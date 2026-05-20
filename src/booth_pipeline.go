// Museum Story Booth — Video Processing Pipeline (Go)
// =====================================================
// Produces a panning photo-reel composite from a raw booth recording.
//
// Build
// -----
//   go mod init booth
//   go get github.com/fsnotify/fsnotify
//   go build -o booth booth_pipeline.go
//
// Usage
// -----
//   ./booth process recording.mp4 ./output --speaker "Jane Smith"
//   ./booth watch   ./watch_folder ./output --title "Community Voices"
//
// Dependencies
// ------------
//   ffmpeg / ffprobe must be on PATH
//   github.com/fsnotify/fsnotify (only needed for 'watch' command)

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
)

// ── Configuration ──────────────────────────────────────────────────────────────

const (
	ffmpegBin  = `C:\Users\x119877\ffmpeg\bin\ffmpeg.exe`
	ffprobeBin = `C:\Users\x119877\ffmpeg\bin\ffprobe.exe`

	frameInterval = 5   // Extract one key frame every N seconds
	panelWidth    = 320 // Each filmstrip panel width in px
	panelHeight   = 240 // Each filmstrip panel height in px
	border        = 8   // White border around each panel in px
	panelGap      = 10  // Horizontal gap between panels in px

	outputWidth  = 1920
	outputHeight = 1080
	outputFPS    = 25

	panSpeed     = 55   // Pixels/second the reel scrolls
	subjectScale = 0.65 // Subject height as fraction of outputHeight

	grainStrength = 16
	vignetteAngle = "PI/4"

	defaultExhibit = "Stories from the Community"
	fontBold       = `../lib/fonts/DejaVuSans-Bold.ttf`
	fontRegular    = `../lib/fonts/DejaVuSans.ttf`
)

// videoExtensions that trigger the watch pipeline
var videoExtensions = map[string]bool{
	".mp4": true, ".mov": true, ".mxf": true, ".avi": true,
}

// ── Metadata ──────────────────────────────────────────────────────────────────

type Metadata struct {
	ContentID       string  `json:"content_id"`
	Exhibit         string  `json:"exhibit"`
	Speaker         string  `json:"speaker"`
	DateRecorded    string  `json:"date_recorded"`
	DurationSeconds float64 `json:"duration_seconds"`
	PipelineVersion string  `json:"pipeline_version"`
}

// ── Helpers ────────────────────────────────────────────────────────────────────

// run executes a command, streaming its stderr to our log. Fatal on error.
func run(label string, name string, args ...string) {
	if label != "" {
		log.Printf("  %s", label)
	}
	preview := name + " " + strings.Join(args, " ")
	if len(preview) > 120 {
		preview = preview[:120] + "…"
	}
	log.Printf("  $ %s", preview)

	cmd := exec.Command(name, args...)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("command failed: %v", err)
	}
}

// probeFormat returns the "format" section from ffprobe as a map.
func probeFormat(path string) map[string]interface{} {
	out, err := exec.Command(
		ffprobeBin, "-v", "quiet", "-print_format", "json",
		"-show_format", "-show_streams", path,
	).Output()
	if err != nil {
		log.Fatalf("ffprobe failed: %v", err)
	}
	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		log.Fatalf("ffprobe JSON parse: %v", err)
	}
	return result
}

func getDuration(path string) float64 {
	info := probeFormat(path)
	format := info["format"].(map[string]interface{})
	d, _ := strconv.ParseFloat(format["duration"].(string), 64)
	return d
}

func getVideoSize(path string) (int, int) {
	info := probeFormat(path)
	streams := info["streams"].([]interface{})
	for _, s := range streams {
		stream := s.(map[string]interface{})
		if stream["codec_type"] == "video" {
			w := int(stream["width"].(float64))
			h := int(stream["height"].(float64))
			return w, h
		}
	}
	log.Fatalf("no video stream in %s", path)
	return 0, 0
}

// escDrawtext escapes characters special to ffmpeg's drawtext filter.
func escDrawtext(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `'`, `\'`)
	s = strings.ReplaceAll(s, `:`, `\:`)
	return s
}

// glob returns sorted matches, fatal if the pattern errors.
func glob(pattern string) []string {
	matches, err := filepath.Glob(pattern)
	if err != nil {
		log.Fatalf("glob %s: %v", pattern, err)
	}
	sort.Strings(matches)
	return matches
}

// ── Stage 1: Extract key frames ───────────────────────────────────────────────

func extractFrames(inputPath, framesDir string) []string {
	log.Println("\n[1/5] Extracting key frames...")

	if err := os.MkdirAll(framesDir, 0755); err != nil {
		log.Fatalf("mkdir frames: %v", err)
	}

	vf := fmt.Sprintf(
		"fps=1/%d,scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d",
		frameInterval,
		panelWidth, panelHeight,
		panelWidth, panelHeight,
	)
	run("", ffmpegBin, "-y", "-i", inputPath,
		"-vf", vf, "-q:v", "2",
		filepath.Join(framesDir, "frame_%04d.jpg"),
	)

	frames := glob(filepath.Join(framesDir, "frame_*.jpg"))
	log.Printf("   → %d frames extracted", len(frames))
	return frames
}

// ── Stage 2: Build filmstrip image ────────────────────────────────────────────

func buildFilmstrip(frames []string, tmpDir string) (stripPath string, stripW, stripH int) {
	log.Println("\n[2/5] Building filmstrip image...")

	if len(frames) == 0 {
		log.Fatal("no frames to stitch — check your input video")
	}

	n       := len(frames)
	panelW  := panelWidth  + border*2
	panelH  := panelHeight + border*2
	stripW   = n*panelW + (n-1)*panelGap
	stripH   = panelH
	stripPath = filepath.Join(tmpDir, "filmstrip.png")

	// Build the ffmpeg command inputs and filter_complex string
	args := []string{"-y"}
	for _, f := range frames {
		args = append(args, "-i", f)
	}

	sepia := "colorchannelmixer=" +
		"rr=0.393:rg=0.769:rb=0.189:" +
		"gr=0.349:gg=0.686:gb=0.168:" +
		"br=0.272:bg=0.534:bb=0.131"

	var filterParts []string

	// Per-frame: scale → white border pad → sepia
	for i := range frames {
		filterParts = append(filterParts, fmt.Sprintf(
			"[%d:v]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,pad=%d:%d:%d:%d:color=white,%s[p%d]",
			i,
			panelWidth, panelHeight,
			panelWidth, panelHeight,
			panelW, panelH, border, border,
			sepia, i,
		))
	}

	// Add right-side gap to each panel (except last), then hstack
	for i := 0; i < n; i++ {
		extra := 0
		if i < n-1 {
			extra = panelGap
		}
		filterParts = append(filterParts, fmt.Sprintf(
			"[p%d]pad=%d:%d:0:0:color=0x1a1a1a[g%d]",
			i, panelW+extra, panelH, i,
		))
	}

	var stackInputs strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&stackInputs, "[g%d]", i)
	}
	filterParts = append(filterParts,
		fmt.Sprintf("%shstack=inputs=%d[strip]", stackInputs.String(), n),
	)

	args = append(args,
		"-filter_complex", strings.Join(filterParts, ";"),
		"-map", "[strip]",
		"-frames:v", "1",
		stripPath,
	)

	run("Stitching panels with hstack…", ffmpegBin, args...)
	return stripPath, stripW, stripH
}

// ── Stage 3: Composite ────────────────────────────────────────────────────────

func composite(
	inputPath, stripPath string,
	stripW, stripH int,
	outputPath string,
	meta map[string]string,
) {
	log.Println("\n[3/5] Compositing panning reel + subject video...")

	duration       := getDuration(inputPath)
	srcW, srcH     := getVideoSize(inputPath)

	// Subject dimensions, centred
	subjH := int(float64(outputHeight) * subjectScale)
	subjW := int(float64(subjH) * float64(srcW) / float64(srcH))
	subjX := (outputWidth  - subjW) / 2
	subjY := (outputHeight - subjH) / 2

	// Tile copies to cover full pan travel
	totalTravel := int(duration*panSpeed) + outputWidth
	tileCopies  := totalTravel/stripW + 2
	if tileCopies < 2 {
		tileCopies = 2
	}

	stripY := (outputHeight - stripH) / 2
	panX   := fmt.Sprintf("-(t*%d)", panSpeed)

	// Drawtext for title card
	title   := escDrawtext(meta["title"])
	date    := escDrawtext(meta["date"])
	speaker := escDrawtext(meta["speaker"])

	drawtext := fmt.Sprintf(
		"drawtext=fontfile=%s:text='%s':fontcolor=white:fontsize=28:alpha=0.85:x=(w-text_w)/2:y=h-70:shadowcolor=black:shadowx=1:shadowy=1,"+
			"drawtext=fontfile=%s:text='%s':fontcolor=0xdddddd:fontsize=20:alpha=0.7:x=(w-text_w)/2:y=h-36:shadowcolor=black:shadowx=1:shadowy=1",
		fontBold, title,
		fontRegular, date,
	)
	if speaker != "" {
		drawtext += fmt.Sprintf(
			",drawtext=fontfile=%s:text='%s':fontcolor=0xffd580:fontsize=22:alpha=0.90:x=(w-text_w)/2:y=30:shadowcolor=black:shadowx=1:shadowy=1",
			fontRegular, speaker,
		)
	}

	filterComplex := strings.Join([]string{
		fmt.Sprintf("[1:v]tile=%dx1[strip_tiled]", tileCopies),
		fmt.Sprintf("[strip_tiled]crop=%d:%d:'%s':0[reel]", outputWidth, stripH, panX),
		fmt.Sprintf("color=c=0x111111:s=%dx%d:r=%d[bg]", outputWidth, outputHeight, outputFPS),
		fmt.Sprintf("[bg][reel]overlay=0:%d[bg_reel]", stripY),
		fmt.Sprintf("[0:v]scale=%d:%d[subject]", subjW, subjH),
		fmt.Sprintf("[bg_reel][subject]overlay=%d:%d[with_subject]", subjX, subjY),
		fmt.Sprintf("[with_subject]noise=alls=%d:allf=t+u[grainy]", grainStrength),
		fmt.Sprintf("[grainy]vignette=angle=%s:mode=forward[vignetted]", vignetteAngle),
		fmt.Sprintf("[vignetted]%s[out]", drawtext),
	}, ";")

	run("Rendering composite (this may take a while)…",
		ffmpegBin, "-y",
		"-i", inputPath,
		"-i", stripPath,
		"-filter_complex", filterComplex,
		"-map", "[out]",
		"-map", "0:a",
		"-c:v", "libx264", "-preset", "fast", "-crf", "18",
		"-c:a", "aac", "-b:a", "192k",
		"-t", strconv.FormatFloat(duration, 'f', 3, 64),
		"-r", strconv.Itoa(outputFPS),
		outputPath,
	)
}

// ── Stage 4: Archive copy ─────────────────────────────────────────────────────

func archiveCopy(displayPath, archivePath string) {
	log.Println("\n[4/5] Writing archive copy (ProRes HQ)...")
	run("",
		ffmpegBin, "-y",
		"-i", displayPath,
		"-c:v", "prores_ks", "-profile:v", "3",
		"-c:a", "pcm_s16le",
		archivePath,
	)
}

// ── Stage 5: Sidecar metadata ─────────────────────────────────────────────────

func writeSidecar(outputPath string, meta map[string]string, duration float64) {
	log.Println("\n[5/5] Writing sidecar metadata...")

	stem    := strings.TrimSuffix(filepath.Base(outputPath), filepath.Ext(outputPath))
	exhibit := meta["title"]
	if exhibit == "" {
		exhibit = defaultExhibit
	}

	data := Metadata{
		ContentID:       stem,
		Exhibit:         exhibit,
		Speaker:         meta["speaker"],
		DateRecorded:    meta["date"],
		DurationSeconds: math_round(duration, 2),
		PipelineVersion: "1.0",
	}

	sidecarPath := strings.TrimSuffix(outputPath, filepath.Ext(outputPath)) + ".json"
	b, _ := json.MarshalIndent(data, "", "  ")
	if err := os.WriteFile(sidecarPath, b, 0644); err != nil {
		log.Fatalf("write sidecar: %v", err)
	}
	log.Printf("   → %s", sidecarPath)
}

// math_round rounds f to decimalPlaces (avoids importing math just for this).
func math_round(f float64, places int) float64 {
	pow := 1.0
	for i := 0; i < places; i++ {
		pow *= 10
	}
	return float64(int(f*pow+0.5)) / pow
}

// ── Orchestrator ──────────────────────────────────────────────────────────────

func process(inputPath, outputDir string, meta map[string]string) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("mkdir output: %v", err)
	}
	if meta == nil {
		meta = map[string]string{}
	}
	if meta["title"] == "" {
		meta["title"] = defaultExhibit
	}
	if meta["date"] == "" {
		meta["date"] = time.Now().Format("January 02, 2006")
	}

	stem       := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	displayOut := filepath.Join(outputDir, stem+"_display.mp4")
	archiveOut := filepath.Join(outputDir, stem+"_archive.mov")

	log.Printf("\n%s", strings.Repeat("=", 60))
	log.Printf("  Processing: %s", filepath.Base(inputPath))
	log.Printf("%s", strings.Repeat("=", 60))

	// Use a temp dir for intermediate files; cleaned up automatically
	tmpDir, err := os.MkdirTemp("", "booth-*")
	if err != nil {
		log.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	frames := extractFrames(inputPath, filepath.Join(tmpDir, "frames"))
	stripPath, stripW, stripH := buildFilmstrip(frames, tmpDir)
	composite(inputPath, stripPath, stripW, stripH, displayOut, meta)
	writeSidecar(displayOut, meta, getDuration(inputPath))
	archiveCopy(displayOut, archiveOut)

	log.Printf("\n✓  Display copy  → %s", displayOut)
	log.Printf("✓  Archive copy  → %s", archiveOut)
}

// ── Watchdog ──────────────────────────────────────────────────────────────────

func watch(watchDir, outputDir, defaultTitle string) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatalf("fsnotify: %v", err)
	}
	defer watcher.Close()

	if err := watcher.Add(watchDir); err != nil {
		log.Fatalf("watch dir: %v", err)
	}
	log.Printf("Watching %q for new recordings…  (Ctrl+C to stop)", watchDir)

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			if event.Op&fsnotify.Create == 0 {
				continue
			}
			ext := strings.ToLower(filepath.Ext(event.Name))
			if !videoExtensions[ext] {
				continue
			}

			// Wait a moment for the camera software to finish writing
			time.Sleep(4 * time.Second)

			log.Printf("\n★  New recording detected: %s", filepath.Base(event.Name))
			go func(path string) {
				process(path, outputDir, map[string]string{
					"title": defaultTitle,
					"date":  time.Now().Format("January 02, 2006"),
				})
			}(event.Name)

		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			log.Printf("watcher error: %v", err)
		}
	}
}

// ── CLI ───────────────────────────────────────────────────────────────────────

func main() {
	log.SetFlags(0) // cleaner output without timestamps

	processCmd := flag.NewFlagSet("process", flag.ExitOnError)
	pTitle     := processCmd.String("title",   defaultExhibit, "Exhibit name")
	pSpeaker   := processCmd.String("speaker", "",             "Speaker name (optional)")

	watchCmd   := flag.NewFlagSet("watch", flag.ExitOnError)
	wTitle     := watchCmd.String("title", defaultExhibit, "Exhibit name")

	if len(os.Args) < 2 {
		fmt.Println("Usage:")
		fmt.Println("  booth process <input> <output_dir> [--title ...] [--speaker ...]")
		fmt.Println("  booth watch   <watch_dir> <output_dir> [--title ...]")
		os.Exit(1)
	}

	switch os.Args[1] {

	case "process":
		processCmd.Parse(os.Args[4:])
		if len(os.Args) < 4 {
			fmt.Println("Usage: booth process <input> <output_dir>")
			os.Exit(1)
		}
		process(os.Args[2], os.Args[3], map[string]string{
			"title":   *pTitle,
			"speaker": *pSpeaker,
			"date":    time.Now().Format("January 02, 2006"),
		})

	case "watch":
		watchCmd.Parse(os.Args[4:])
		if len(os.Args) < 4 {
			fmt.Println("Usage: booth watch <watch_dir> <output_dir>")
			os.Exit(1)
		}
		watch(os.Args[2], os.Args[3], *wTitle)

	default:
		fmt.Printf("Unknown command %q. Use 'process' or 'watch'.\n", os.Args[1])
		os.Exit(1)
	}
}