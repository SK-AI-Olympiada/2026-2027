# Chyby v oceli

V oceliarňach sa plechy kontrolujú kamerovými systémami priamo na výrobnej linke. Každá nájdená chyba na povrchu – škrabanec, škvrna, nečistota či vydutie – sa musí rýchlo zaradiť do správnej kategórie, pretože rôzne chyby majú rôzne príčiny a rôzne dôsledky: niektoré znamenajú len kozmetickú chybu, iné vyradenie plechu alebo zastavenie linky. Operátor pritom potrebuje vedieť nielen, čo systém rozhodol, ale aj prečo – aby mohol odhaliť chybu v nastavení stroja a aby modelu dôveroval.

Vašou úlohou je pomôcť oceliarni vytvoriť model, ktorý z nameraných vlastností chyby určí, o aký typ chyby ide.

## Formát dátových vzoriek

Každý riadok datasetu opisuje jednu chybu nájdenú na povrchu oceľového plechu pomocou 27 numerických príznakov získaných obrazovou analýzou: poloha a rozmery chyby, obvod, svietivosť, typ ocele, hrúbka plechu a rôzne tvarové indexy.

Chyby patria do 7 typov: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults. Triedy sú výrazne nevyvážené.

- **train.csv** – 1 552 chýb. Obsahuje stĺpec `id` (od 1 po 1552), 27 príznakov a stĺpec `fault_type` so správnym typom chyby.
- **test.csv** – 389 chýb. Obsahuje stĺpec `id` (od 1553 po 1941) a 27 príznakov, typ chyby chýba. Identifikátory používajte presne tak, ako sú v súbore.

Stĺpce `TypeOfSteel_A300` a `TypeOfSteel_A400` sú binárne a určujú typ ocele: každý plech je práve jedného z týchto dvoch typov.

## Podúlohy

### Podúloha 1 – Automatický triedič chýb (100 bodov)

Natrénujte model strojového učenia, ktorý pre každú chybu z `test.csv` určí jej typ. Môžete použiť ľubovoľný model aj vlastné odvodené príznaky.

Keďže triedy sú nevyvážené, model sa nehodnotí presnosťou (accuracy), ale metrikou weighted F1-score: F1-score sa vypočíta pre každý zo 7 typov chýb zvlášť a výsledkom je ich vážený priemer. Málo početné triedy (napr. Dirtiness) tak majú rovnakú váhu ako najčastejšie triedy.

## Výstupný formát

Výstupný formát je .csv a musí obsahovať tri stĺpce:

| subtaskID | datapointID | Answer |
|-----------|-------------|--------|
| podúloha (1) | ID z testovacej množiny | odpoveď pre každé ID |

Posielate jediný .csv súbor.

## Linky a priložené súbory

- Testovacia platforma: <https://judge.nitro-ai.org/competitions/slovenska-olympiada/domaca-priprava/3/view>
- Trénovacie dáta: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/chyby_v_oceli/train_data.csv>
- Testovacie dáta: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/chyby_v_oceli/test_data.csv>
- Príklad výstupu: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/chyby_v_oceli/sample_output.csv>
