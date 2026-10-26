# Zakliata reč

## Príbeh

Na povale starej horárne pod Tatrami našli pri oprave strechy drevenú
truhlicu. Ležali v nej tisícky popísaných lístočkov. Na každom bola jedna
veta po slovensky a pod ňou tá istá veta v jazyku, ktorý nikto nepoznal.

Miestny kronikár tvrdí, že ide o *zakliatu reč*, jazyk, ktorým sa vraj
kedysi dorozumievali víly, aby im ľudia nerozumeli. Podľa povesti sa
kliatba zlomí, keď niekto dokáže prekladať tam aj späť bez jedinej chyby.

Jazykovedci z univerzity sa na lístočky pozreli a pokrčili plecami. Slová
znejú mäkko a melodicky, napríklad *„Tatakan muman per ripazano.“*, no
v žiadnom slovníku ich nenájdeš. Zdá sa, že jazyk má vlastnú gramatiku:
slová sa menia podľa toho, kto čo robí a kedy, a vety sú poskladané v
inom poradí ako v slovenčine.

Lístočkov je však dosť na to, aby sa jazyk dal **naučiť z príkladov**. A presne
to dokážu prekladové modely. Tvojou úlohou je postaviť
prekladač zo zakliatej reči do slovenčiny aj naopak.

> **Dôležité pri odovzdávaní:** Spolu so súborom `submission.csv` musíš
> odovzdať aj svoj zdrojový kód (najviac 35 KB), ktorý tento súbor
> vyrobil. Kód musí byť reprodukovateľný. Pri tréningu uveď aj príkaz, verzie
> závislostí a seed. Z notebooku odstráň výstupy a neprikladaj váhy modelu.
> Ak používaš doplnený `train.py`, odovzdaj aj `starter.py`.
> Ak model trénuješ, odovzdaný kód musí obsahovať aj tréning.


## Opis dátových vzoriek

Dátové súbory sú v priečinku `data/`, majú formát CSV a kódovanie UTF-8.
Polia s čiarkou alebo úvodzovkami sú uzavreté v úvodzovkách. Je to
štandardný formát CSV, preto na čítanie použi modul `csv` alebo `pandas`,
nie `line.split(",")`.

**`train.csv`** -- 11 378 trénovacích párov viet, stĺpce:

| id | zak | sk |
|----|-----|----|
| t00069 | Tatakan muman per ripazano. | Dievčatá sa tešia na gymnastiku. |
| t00025 | Talen hon helu deso mohe. | Cez deň mráz trochu poľaví. |

**`test.csv`** -- 1 400 viet na preklad (700 pre každú podúlohu), stĺpce:

| subtaskID | datapointID | input |
|-----------|-------------|-------|
| 1 | 0 | veta v zakliatej reči |
| 2 | 0 | veta po slovensky |

Pre podúlohu 1 je `input` v zakliatej reči, pre podúlohu 2 po slovensky.
Testovacie vety sa v trénovacích dátach nenachádzajú, ale skoro všetky
slová (ich základné tvary) sa v nich niekde objavia.

**`sample_output.csv`** -- ukážka správneho formátu odovzdania (obsahuje
len skopírované vstupy, nie preklady).

Korene vlastných mien a čísla sa neprekladajú. Mená môžu mať v zakliatej
reči pádovú príponu. Interpunkcia sa zachováva.

V archíve `starter_kit.zip` dostaneš `starter.py`, `starter.ipynb`
a `train.py`. Slovník v `starter.py` sa z dát naučí, ktoré slová si
v pároch zodpovedajú, a vyrobí platný `submission.csv`. V notebooku vyhodnotíš slovník
a doplníš tréningový kód pre malý model. `train.py` je kostra programu, v ktorej doplníš tri
funkcie: vytvoriť príklady pre oba smery, doplniť krok tréningu a vybrať
stav modelu (checkpoint) podľa výsledku na validačných pároch. Bez ich doplnenia model nenatrénuješ.
Na prezeranie dát, slovníkový preklad aj kontrolu funkcií stačí CPU.
Na tréning modelu použi Colab s GPU T4.


## Podúlohy

### Podúloha 1 -- Zo zakliatej reči do slovenčiny (50 bodov)

Prelož každú vetu podúlohy 1 z `test.csv` do slovenčiny. Pozor, jeden
tvar v zakliatej reči môže zodpovedať viacerým slovenským tvarom. Tvoj
preklad musí byť správne vyskloňovaný a vyčasovaný.

### Podúloha 2 -- Zo slovenčiny do zakliatej reči (50 bodov)

Prelož každú vetu podúlohy 2 z `test.csv` do zakliatej reči. Tu musíš
z kontextu slovenskej vety odhadnúť, aké prípony má slovo v zakliatej
reči dostať a kam vo vete patrí.

Pravidlá zakliatej reči môžeš odhaliť z trénovacích dát.
Každé základné slovo má jeden koreň. Tvary a poradie slov závisia
od gramatiky vety. Môžeš skúsiť slovník, model alebo ich kombináciu
a na validačných dátach overiť, čo funguje lepšie.

Pri preklade do slovenčiny môže byť správnych viac znení.
Niektoré rozdiely, napríklad rod v minulom čase, zakliata reč
nerozlišuje. chrF porovnáva výsledok s jedným referenčným prekladom,
preto pri rozbore chýb sleduj aj význam vety.

Čo môžeš vyskúšať:

- natrénovať vlastný malý Transformer (enkodér-dekodér) od nuly, keďže dát je
  pomerne dosť a jazyk je pravidelnejší ako skutočné jazyky,
- dotrénovať predtrénovaný model typu enkodér-dekodér (napr. `google/mt5-small`),
- jazykový model s príkladmi v prompte (few-shot) -- vyber mu podobné vety
  z trénovacích dát.


## Od prvého pokusu k lepšiemu modelu

Notebook `starter.ipynb` má štyri kroky. Pri každom si zapíš výsledok,
aby si vedel vysvetliť, prečo si v ďalšom pokuse niečo zmenil.

1. **Prezri si dáta.** Prečítaj niekoľko párov a pomenuj jednu vec,
   ktorú preklad slovo po slove nedokáže zachytiť
2. **Zmeraj slovník.** Odlož 500 celých párov na validáciu, uč sa iba
   zo zvyšku a spočítaj chrF pre každý smer zvlášť. Pozri si konkrétne chyby
3. **Doplň tréning a natrénuj model.** V `train.py` doplň `make_examples`,
   `train_batch` a `select_checkpoint`. Príkazom `python train.py --check`
   over všetky tri funkcie. Potom model natrénuj, použi rovnaké validačné
   páry a porovnaj skóre aj chyby so slovníkom. Namiesto `train.py` môžeš
   použiť aj vlastný model
4. **Zmeň jednu vec.** Napríklad dropout, veľkosť modelu alebo tokenizér.
   Ostatné podmienky nechaj rovnaké a zapíš, čo výsledok ukázal

Každý pár viet najprv priraď do trénovacej alebo validačnej časti a až potom
vytvor oba smery prekladu. Validačné vety nepoužívaj na učenie slovníka,
tokenizéra ani modelu. Súťažný test si nechaj na finálne preklady.

Stačí malá tabuľka pokusov s oboma skóre a rozbor troch chýb.
Nebodujú sa, slúžia ako pomôcka pre tvoj postup.
Prvé porovnanie zvládneš bez GPU. Tréningový skript má predvolený časový
limit 20 minút (`--minutes`), ktorý kontroluje po každej epoche.
Na záverečné vyhodnotenie si nechaj čas navyše.


## Výstupný formát

Pre obe podúlohy odovzdávaš **jediný** súbor `.csv` s hlavičkou:

```
subtaskID,datapointID,answer
```

- `answer` je tvoj preklad danej vety (jeden riadok, bez zalomení),
- pre každý riadok `test.csv` jeden riadok s rovnakým `subtaskID` a
  `datapointID`,
- ak preklad obsahuje čiarku, pole musí byť v úvodzovkách, čo za teba urobí modul `csv`

Chýbajúci riadok sa počíta ako prázdny preklad.


## Hodnotenie

Kvalita prekladu sa v každej podúlohe meria metrikou **chrF** (presnejšie
jej korpusovou verziou z knižnice `sacrebleu`, `CHRF()` s predvolenými
nastaveniami). chrF porovnáva tvoj preklad so správnym prekladom po
skupinkách znakov (n-gramoch), takže sa čiastočne započíta aj
slovo so správnym koreňom a nesprávnou koncovkou. Skóre je v rozsahu 0 až 100.

Každá podúloha má najviac 50 bodov. Medzi dolnou a hornou hranicou
sa chrF lineárne prepočíta na 5 až 45 bodov. Body sa zaokrúhľujú na celé
čísla. Neplatné odovzdania získajú 0 bodov a nezapočítavajú sa do limitu
odovzdaní. Presné hranice sú v tabuľke.

| Podúloha | chrF pod hranicou | medzi hranicami | chrF ≥ 78 |
|----------|-------------------|-----------------|-----------|
| 1 | chrF < 49: 2 body | 49 ≤ chrF < 78: 5 -- 45 bodov | 50 bodov |
| 2 | chrF < 50: 2 body | 50 ≤ chrF < 78: 5 -- 45 bodov | 50 bodov |

Spolu môžeš získať **100 bodov**. `starter.py` dosiahne chrF približne
48,5 v podúlohe 1 a 51,4 v podúlohe 2, spolu 9 bodov. Model porovnávaj
so slovníkom na rovnakých validačných pároch. Doplnený `train.py` po
tréningu uloží aj ukážky chýb do súboru s príponou `.dev.csv`.

## Linky a priložené súbory

Začni otvorením notebooku v Colabe. Do jeho súborov nahraj oba archívy
s dátami a `starter_kit.zip`, potom postupuj podľa buniek v notebooku.

- [Súťažná platforma Nitro AI](https://judge.nitro-ai.org/)
- [Trénovacie dáta (ZIP)](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/train_data.zip)
- [Testovacie dáta (ZIP)](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/test_data.zip)
- [Príklad výstupu (CSV)](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/sample_output.csv)
- [Štartovací balík (ZIP)](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/starter_kit.zip)
- [Otvoriť notebook v Colabe](https://colab.research.google.com/github/SK-AI-Olympiada/2026-2027/blob/main/domaca-priprava/zakliata_rec/starter.ipynb)
- [Stiahnuť notebook](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/starter.ipynb)
- [Štartovací slovník](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/starter.py)
- [Kostra tréningu modelu](https://raw.githubusercontent.com/SK-AI-Olympiada/2026-2027/main/domaca-priprava/zakliata_rec/train.py)
