# Käkisalmi proto

<hr>

## Video Pipeline Implementation

### FFmpeg with Go

Teknisesti monimutkaisempi implementaatio, mutta paljon suorituskyvykkäämpi.

GOOS=linux GOARCH=amd64 go build -o booth_linux booth_pipeline.go
go build -o build/booth_win.exe booth_pipeline.go


### Python with MoviePy

Helpompi muokata ( jos vain osaisi pythonia ;) ), mutta varmaankin hitain kaikista, paitsi ehkä Adobe AE pipeline.


## Big Picture- päätökset

Aiotaanko kuvauspisteeseen tehdä minkäänlaista valmistusta e.g. vihreä seinä, valaistus yms? Tarvitaanko AI taustanpoistoa, vai voidaanko hyödyntää oikeita työkaluja, kuten CorridorKey?

Onko tilaajalla jonkinlainen kuva tarkalleen minkälaista toteutusta haluavat? Onko esim. vierivä kuvatausta hyvä?

Minkälainen palvelin museolla on käytössä tällä hetkellä? Voidaanko sitä hyödyntää, vai hankitaanko paikan päälle kone tätä varten? Jos palvelinta voidaan hyödyntää, onko siinä riittävästi resursseja videon käsittelyyn?


### Notes

Nykyinen koodi käyttää hard-coded polkuja koulukoneelta
Koulun systeemin takia myös riippuvuudet ovat local
