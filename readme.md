# Käkisalmi protoyyppi

<hr>

## Yleiset Ideat

Jos toteutukseen halutaan ulkoasun puolelta vain taustanpoisto ja henkilön asetus toisen taustan päälle ja *ehkä* esim. tausta vaihtuu välillä tai vierii yms., tämän voi varmaankin hyvin toteuttaa suoraan pelkästään ffmpeg:llä tai muulla samanlaisella. Jos halutaan paljon monimutkaisempia efektejä, voidaan siirtyä täysin takaisin AE (After Effects) pipelinen pariin. Tämän osion toteutus ffmpeg:llä ei tietenkään tarkoita, etteikö kumpaakin pipelinea voisi käyttää.

<hr>

## Video Pipeline Implementation

> *Tämänhetkiset prototyypit vain suuntaa antavat. Älä kiitos käytä prod ympäristössä. Kaikki versiot sisältävät runsaasti AI:n käsityötä, joten - kuten viittasin - en suosittele oikeaan implementatioon.*

Muuttujat kielen ja kirjaston valitsemiseen on seuraavat: kuinka paljon kokemusta tekijöillä on kieleen, onko kirjasto helppokäyttöinen/riittävän ominaisuusrikas, ja kuinka suorituskykyinen ohjelman halutaan olevan.

### Rust with FFmpeg

* Teknisesti monimutkaisempi implementaatio kuin Python MoviePy:lla, mutta paljon suorituskyvykkäämpi.

### Go with FFmpeg

* Enimmäkseen mielenkiinnosta mukana, voi olla yhtä hyvä vaihtoehto kuin Rust.

    `GOOS=linux GOARCH=amd64 go build -o booth_linux booth_pipeline.go` <br>
	`go build -o build/booth_win.exe booth_pipeline.go`

### Python with MoviePy

* Helpompi muokata ( jos vain osaisi pythonia ;) ), mutta varmaankin hitain kaikista, paitsi ehkä Adobe AE pipeline.

## Big Picture- päätökset

Aiotaanko kuvauspisteeseen tehdä minkäänlaista valmistusta e.g. vihreä seinä, valaistus yms? Tarvitaanko AI taustanpoistoa, vai voidaanko hyödyntää oikeita työkaluja, kuten CorridorKey?

Onko tilaajalla jonkinlainen kuva tarkalleen minkälaista toteutusta haluavat? Onko esim. vierivä kuvatausta hyvä?

Minkälainen palvelin museolla on käytössä tällä hetkellä? Voidaanko sitä hyödyntää, vai hankitaanko paikan päälle kone tätä varten? Jos palvelinta voidaan hyödyntää, onko siinä riittävästi resursseja videon käsittelyyn?

### Notes

Nykyinen koodi käyttää hard-coded polkuja koulukoneelta ([41a9eb843c](https://git.shambali.org/de4dn0te/kakisalmi/commit/41a9eb843c7435882ad41a606773f6e8083a8534))
