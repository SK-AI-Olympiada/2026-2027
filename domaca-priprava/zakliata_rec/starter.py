"""Zakliata reč -- štartovací kód (slabý baseline).

Spustenie (v priečinku úlohy, kde je zložka data/):
    python starter.py
    python starter.py --validate  # navyše potrebuje sacrebleu

Vyrobí súbor submission.csv vo formáte subtaskID,datapointID,answer.

Čo robí: postaví si "slovník" slovo -> slovo podľa toho, ktoré slová sa
v trénovacích pároch vyskytujú spolu, a vetu preloží slovo po slove
v pôvodnom poradí. Učí sa súvislosti medzi slovami, ale ignoruje kontext
a gramatiku.
Dostaneš s ním len málo bodov -- je to len štart, aby si mal platný súbor.
"""

import argparse
import csv
import random
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(s):
    # Rozdelí vetu na slová a interpunkciu: "Ahoj, svet." -> ["Ahoj", ",", "svet", "."]
    return TOKEN_RE.findall(s)


def detokenize(tokens):
    s = " ".join(tokens)
    return re.sub(r" ([,.!?;:])", r"\1", s)


def is_word(t):
    return any(c.isalpha() for c in t)


def build_dictionary(pairs):
    """pairs = zoznam (zdrojová veta, cieľová veta).

    Pre každé zdrojové slovo nájde cieľové slovo s najvyšším Diceovým
    koeficientom: 2 * (počet viet, kde sú obe) / (počet viet so zdrojovým + počet viet s cieľovým).
    """
    src_count, tgt_count = Counter(), Counter()
    together = defaultdict(Counter)
    for src, tgt in pairs:
        S = {w.lower() for w in tokenize(src) if is_word(w)}
        T = {w.lower() for w in tokenize(tgt) if is_word(w)}
        src_count.update(S)
        tgt_count.update(T)
        for a in S:
            together[a].update(sorted(T))  # sorted -> rovnaký výsledok pri každom spustení
    table = {}
    for a, cands in together.items():
        table[a] = max(sorted(cands), key=lambda b: 2 * cands[b] / (src_count[a] + tgt_count[b]))
    return table


def translate(sentence, table):
    out = []
    for w in tokenize(sentence):
        if is_word(w) and w.lower() in table:
            out.append(table[w.lower()])
        else:
            # interpunkcia, čísla, mená a neznáme slová sa len skopírujú
            out.append(w)

    # POZRI: Slová v zakliatej reči majú zjavne koreň a za ním prípony.
    #        Čo sa stane so slovom, keď je v množnom čísle? V otázke? V zápore?
    # POZRI: Poradie slov nie je rovnaké ako v slovenčine (kde je sloveso?).
    # POZRI: Slovník slovo->slovo nevie vyskloňovať. Skutočný prekladový model
    #        sa môže naučiť aj gramatiku. Pozri kroky v starter.ipynb
    #        a kostru modelu s tromi úlohami v train.py. Výsledok over na validačných pároch.
    if out:
        out[0] = out[0][:1].upper() + out[0][1:]
    return detokenize(out)


def load_pairs(path="data/train.csv"):
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["zak"], r["sk"]) for r in csv.DictReader(f)]


def split_pairs(pairs, dev_size=500, seed=0):
    """Odlož celé páry pred vytvorením oboch smerov prekladu."""
    if not 0 < dev_size < len(pairs):
        raise ValueError("Validačná časť musí byť neprázdna a menšia ako celé dáta.")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    return shuffled[dev_size:], shuffled[:dev_size]


def fit(pairs):
    return {
        1: build_dictionary(pairs),
        2: build_dictionary([(sk, zak) for zak, sk in pairs]),
    }


def evaluate(pairs, tables, n_examples=3):
    """chrF počítame zo skutočne odložených párov, zvlášť pre oba smery."""
    from sacrebleu.metrics import CHRF

    scores = {}
    for st, name in ((1, "zak → sk"), (2, "sk → zak")):
        sources = [z if st == 1 else s for z, s in pairs]
        references = [s if st == 1 else z for z, s in pairs]
        predictions = [translate(s, tables[st]) for s in sources]
        scores[st] = CHRF().corpus_score(predictions, [references]).score
        print(f"{name}: validačné chrF = {scores[st]:.2f}")
        for src, ref, pred in list(zip(sources, references, predictions))[:n_examples]:
            print(f"  Vstup:    {src}\n  Vzor:     {ref}\n  Predikcia: {pred}\n")
    return scores


def write_submission(tables, test_path="data/test.csv", out_path="submission.csv"):
    with open(test_path, encoding="utf-8", newline="") as f:
        test = list(csv.DictReader(f))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["subtaskID", "datapointID", "answer"])
        for r in test:
            st = int(r["subtaskID"])
            w.writerow([st, r["datapointID"], translate(r["input"], tables[st])])
    print(f"Hotovo: {len(test)} riadkov v {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Zmeraj chrF na 500 odložených pároch")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    pairs = load_pairs()
    if args.validate:
        train_pairs, dev_pairs = split_pairs(pairs)
        evaluate(dev_pairs, fit(train_pairs))
    # Po vyhodnotení sa finálny slovník naučí zo všetkých dodaných párov.
    write_submission(fit(pairs), out_path=args.out)


if __name__ == "__main__":
    main()
