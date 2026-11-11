# Træning af vores egne brikker

614 originalbilleder i `../Kryds-og-bolle-billeder/` er matchet med 614 Make Sense-labels i `../labels_my-project-name_2026-11-11-10-08-19/`. Klasse 0 er `kryds`; klasse 1 er `bolle`. Originalerne er bevaret.

Datasættet er klargjort i `egne-brikker/` med **475 træningsbilleder og 139 valideringsbilleder**. Begge grupper indeholder kryds og boller. Træningsscriptet bruger nu dette datasæt frem for lærerens.

Start fra projektets hovedmappe:

```powershell
uv run train-demo/model-train.py
```

Scriptet træner i op til 50 runder og kan stoppe tidligere efter 10 runder uden forbedring. Basismodellen er `train-demo/yolo26n.pt`. Den bedste model gemmes som `train-demo/runs/detect/egne-brikker/weights/best.pt`. Ved senere træninger kan resultatmappen få et nummer; brug stien, træningen udskriver.

Start spillet med den nye model:

```powershell
uv run main.py
```

## Klargøring og opdeling

`prepare_data.py` matcher filnavne og kontrollerer labelkoordinater. Det gemmer træningskopier på højst 1280 pixels på den længste side. Nabofotos i blokke på 20 holdes samlet, og opdelingen fordeles omtrent 80/20 mellem træning og validering. `egne-brikker/split.json` dokumenterer opdelingen. Originaler og lærerens gamle datasæt ændres ikke.

Datasættet er allerede oprettet. Klargøringsscriptet afviser at overskrive en eksisterende `egne-brikker`-mappe. Træningskopier og modelvægte er ikke med i Git.

Billederne indeholder næsten udelukkende én brik ad gangen. Test derfor modellen med flere brikker og på det tomme 3 × 3-bræt fra USB-kameraet. Tag nye optagelser til denne sluttest; validering fra samme fotosession kan være lettere end det faktiske spil. Tilføj senere billeder med tomt bræt og blandede brikker, hvis genkendelsen giver fejl.
