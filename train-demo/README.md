# Træning af vores egne brikker

De nye originalbilleder ligger i `../Kryds-der-boller/`, og Make Sense-labels ligger i `../Kryds-der-boller-data/`. Alle **34 billeder** har matchende, gyldige labels. Klasse 0 er `kryds`; klasse 1 er `bolle`. I alt er der 170 markerede kryds og 136 markerede boller.

Datasættet er gjort klar i `nye-brikker/`:

- **28 træningsbilleder:** 140 kryds og 112 boller.
- **6 valideringsbilleder:** 30 kryds og 24 boller.

Originalerne, det tidligere datasæt `egne-brikker/` og tidligere modeller er bevaret.

## Start træningen

Kør fra projektets hovedmappe:

```powershell
uv run train-demo/model-train.py
```

Scriptet bruger `nye-brikker/data.yaml` og træner i op til 50 runder. Det kan stoppe efter 10 runder uden forbedring. Basismodellen er `train-demo/yolo26n.pt`. Resultatet gemmes under `train-demo/runs/detect/nye-brikker/`; den bedste model er `weights/best.pt`.

Ved gentagne træninger kan resultatmappen få et nummer. Brug den sti, træningen udskriver.

Start spillet med den nye model, når træningen er færdig:

```powershell
uv run main.py --model "train-demo/runs/detect/nye-brikker/weights/best.pt"
```

Indtil da bruger normal opstart fortsat den tidligere trænede model.

## Klargøring og kontrol

`prepare_data.py` matcher filnavne og kontrollerer klassenumre og koordinater. Det gemmer træningskopier på højst 1280 pixels på den længste side. Kamerafotos fra samme 20 sekunder og kopier med samme tidsstempel holdes samlet. Opdelingen dokumenteres i `nye-brikker/split.json`.

Datasættet er allerede oprettet. Klargøringsscriptet afviser at overskrive en eksisterende outputmappe. For andre billedmapper kan I bruge:

```powershell
uv run train-demo/prepare_data.py --images "BILLEDMAPPE" --labels "LABELMAPPE" --output "train-demo/nyt-datasaet"
```

De nye billeder indeholder alle ni brikker. Der er kun 34 billeder fra en kort fotosession, så valideringen fortæller ikke alene, hvor godt modellen virker i hele spillet. Test med nye optagelser fra USB-kameraet, tomt bræt og forskellige antal brikker. Tilføj billeder af disse situationer, hvis genkendelsen fejler.

Det tidligere datasæt med 614 billeder findes fortsat i `egne-brikker/`. Det er ikke blandet sammen med de nye billeder.
