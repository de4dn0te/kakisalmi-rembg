module booth

go 1.25.0

require github.com/fsnotify/fsnotify v1.7.0

require golang.org/x/sys v0.35.0 // indirect

// Replace directives point the module resolver to your local copies.
// Paths are relative to this go.mod file — adjust if your layout differs.
replace (
	github.com/fsnotify/fsnotify => ../lib/fsnotify
	golang.org/x/sys => ../lib/sys
)
