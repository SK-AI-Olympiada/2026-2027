# Krajinky

Niekde medzi zasneženými vrcholmi hôr, pokojnou hladinou jazera a lúkou zaliatou večerným slnkom sa nachádza krajina, do ktorej sa nikdy nevyberiete. Nevedie k nej žiadna cesta a nenájdete ju na žiadnej mape. Vznikla totiž v predstavivosti generatívneho modelu.

Takýchto krajiniek sme si nechali vytvoriť celú zbierku. Niektoré by sa vynímali na pohľadnici, iné nad pohovkou v obývačke a pri ďalších by človek najradšej hneď zbalil batoh. Našťastie ich netreba nikam vešať ani kvôli nim cestovať. Stačí ich správne uložiť na disk a môžete sa nimi kochať celý deň.

Lenže práve pri ukladaní sa nášmu digitálnemu krajinárovi prihodila nepríjemnosť. Trénovacie obrázky ešte uložil správne, no pri testovacích už niečo pokazil. Súbory síce existujú, obrázky sa dajú načítať, ale krajinky nevyzerajú tak, ako mali. Či za tým bola nepozornosť, únava alebo priveľa obdivovania vlastnej tvorby, už nezistíme.

Zostala vám zbierka správne uložených trénovacích obrázkov a zbierka testovacích obrázkov, ktoré potrebujú pomoc. Preskúmajte ich, odhaľte, čo sa pri ukladaní pokazilo, a pokúste sa testovacím krajinkám vrátiť ich pôvodnú podobu.

## Formát dátových vzoriek

Všetky dátové vzorky v tejto úlohe majú rovnaký formát. Sú uložené ako .csv s tromi stĺpcami. Prvý je číslo podúlohy, druhý číslo obrázka a tretí je kód obrázka. Ako sa obrázky kódujú a dekódujú, je možné vidieť v starter_kit.py

## Podúlohy

### Podúloha 1 - Náprava (100 bodov)

Vašou úlohou je opraviť testovacie obrázky tak, aby sa čo najviac podobali na svoju správnu podobu.

Trénovacie obrázky vám poslúžia ako ukážka toho, ako mala zbierka vyzerať pred nešťastnou chybou. Spôsob nápravy je už na vás. Dôležité je, aké krajinky napokon odovzdáte.

Kvalita opravy sa vyhodnocuje pomocou strednej kvadratickej chyby (MSE) medzi odovzdanými obrázkami a ich správnymi podobami. Čím viac sa opravené testovacie dáta podobajú svojej správnej podobe, tým viac bodov získate. Systém hodnotenia je nastavený tak, že bez zmeny je to za nula bodov.

## Výstupný formát

Výstupný formát je .csv a musí obsahovať tri stĺpce:

| **subtaskID** | **datapointID**       | **Answer**  |
|---------------|-----------------------|-------------|
| podúloha (1)  | číslo obrázka (0–799) | kód obrázka |

Posielate jediný .csv súbor.

## Linky a priložené súbory

Testovacia platforma: <https://judge.nitro-ai.org/competitions/slovenska-olympiada/domaca-priprava/1/view>

Trénovacie dáta: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/krajinky/train_data.csv>

Testovacie dáta: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/krajinky/test_data.csv>

Príklad výstupu: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/krajinky/sample_output.csv>

Starter Kit: <https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/krajinky/starter_kit.pyTraining>
