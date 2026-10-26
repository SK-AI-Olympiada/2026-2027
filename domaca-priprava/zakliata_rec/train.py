"""Zakliata reč: malý Transformer pre oba smery prekladu.

Na Colabe najprv spusti: %pip install tokenizers sacrebleu
Najprv doplň tri úlohy a over ich: python train.py --check
Potom trénuj: python train.py --data data --out submission_neural.csv

Validačné páry zostávajú mimo tréningu aj pri tvorbe výsledného odovzdania.
Skript uloží aj submission_neural.dev.csv s chybami a
submission_neural.metrics.json so skóre. Mená vychádzajú z argumentu --out.
Bez doplnenia funkcií make_examples, train_batch a select_checkpoint
sa tréning nespustí. Slovníkový starter.py funguje aj bez ich doplnenia.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn
from starter import load_pairs, split_pairs


def make_tokenizer(pairs, vocab):
    """Slovník tokenizéra sa učí iba z trénovacej časti."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.train_from_iterator(
        (sentence for pair in pairs for sentence in pair),
        trainers.BpeTrainer(vocab_size=vocab, special_tokens=[
            "<pad>", "<unk>", "<s>", "</s>", "<2sk>", "<2zak>"]),
    )
    return tokenizer


class Seq2Seq(nn.Module):
    def __init__(self, vocab, pad_id, d, heads, layers, dropout, max_len):
        super().__init__()
        self.pad_id = pad_id
        self.d = d
        self.embedding = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.position = nn.Embedding(max_len + 2, d)
        self.transformer = nn.Transformer(
            d, heads, layers, layers, 4 * d, dropout,
            batch_first=True, norm_first=True,
        )
        self.output = nn.Linear(d, vocab, bias=False)
        self.output.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=d ** -0.5)

    def embed(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        return self.embedding(tokens) * math.sqrt(self.d) + self.position(positions)

    def encode(self, source):
        return self.transformer.encoder(
            self.embed(source), src_key_padding_mask=source == self.pad_id)

    def decode(self, target, memory, source_padding):
        # Pri tréningu nesmie dekodér vidieť budúce cieľové tokeny.
        mask = torch.triu(torch.ones(
            target.size(1), target.size(1), device=target.device, dtype=torch.bool), 1)
        hidden = self.transformer.decoder(
            self.embed(target), memory, tgt_mask=mask,
            tgt_key_padding_mask=target == self.pad_id,
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)


def make_examples(pairs, encode, tags):
    """Úloha 1/3: z párov (zak, sk) vytvor príklady pre oba smery.

    Vráť zoznam dvojíc (vstupné tokeny, cieľové tokeny).
    Na začiatok vstupu pridaj značku cieľového jazyka: tags[1] pre
    slovenčinu a tags[2] pre zakliatu reč. Vety zakóduj funkciou encode.
    Cieľ zatiaľ neobsahuje BOS ani EOS. Validácia sem nepatrí.
    """
    raise NotImplementedError("Úloha 1/3: doplň make_examples v train.py")


def apply_gradients(loss, model, optimizer, scaler):
    """Hotová pomôcka pre spätný priechod aj pri zmiešanej presnosti na GPU."""
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()


def train_batch(model, source, target, optimizer, criterion, scaler, use_amp):
    """Úloha 2/3: vykonaj jeden krok učenia a vráť loss ako číslo.

    source aj target sú tenzory [počet viet, dĺžka]. target už má BOS,
    EOS aj výplň. BOS označuje začiatok vety, EOS jej koniec.
    Výplň vyrovnáva dĺžky viet v dávke. Dekodér má z predchádzajúcich tokenov predpovedať
    nasledujúce. Rozhodni, ktorú časť target mu dáš na vstup a ktorú
    použiješ ako správnu odpoveď. Enkodér spracuje source a maska
    source == model.pad_id označuje výplň vstupu.

    criterion už ignoruje výplň. Pre výpočet loss potrebuje skóre
    tvaru [počet tokenov, veľkosť slovníka] a správne ID tvaru [počet tokenov].
    Vynuluj gradienty, vypočítaj loss a zavolaj apply_gradients.
    Na GPU počítaj dopredný priechod v kontexte
    torch.autocast("cuda", dtype=torch.float16, enabled=use_amp).
    Spätný priechod nechaj mimo tohto kontextu.
    """
    raise NotImplementedError("Úloha 2/3: doplň train_batch v train.py")


def select_checkpoint(model, scores, epoch, best_score, best_weights, best_epoch):
    """Úloha 3/3: vráť (najlepšie skóre, uložené váhy, ich epocha).

    scores[1] a scores[2] sú validačné chrF pre oba smery.
    Porovnávaj ich priemer s best_score. Pri zlepšení ulož nezávislú
    kópiu váh modelu, aby ju ďalší tréning nezmenil. Inak vráť doterajší
    výsledok. Pri zhode si nechaj skorší checkpoint.
    """
    raise NotImplementedError("Úloha 3/3: doplň select_checkpoint v train.py")


def check_exercises():
    """Malé kontroly na CPU. Nenahrádzajú porovnanie na validácii."""
    def examples_check():
        examples = make_examples([("zak", "sk")],
                                 lambda text: {"zak": [6, 7], "sk": [8]}[text],
                                 {1: 4, 2: 5})
        assert sorted(examples) == sorted([([4, 6, 7], [8]), ([5, 8], [6, 7])]), \
            "Skontroluj oba smery, značky cieľového jazyka a cieľové tokeny."

    def batch_check():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = Seq2Seq(12, 0, 8, 2, 1, 0.0, 8)
            source = torch.tensor([[4, 6, 0], [5, 8, 9]])
            target = torch.tensor([[2, 8, 3, 0], [2, 6, 7, 3]])
            before = [p.detach().clone() for p in model.parameters()]
            decoder_inputs, labels_seen = [], []
            decode = model.decode
            def record_decode(tokens, *args, **kwargs):
                decoder_inputs.append(tokens.detach().clone())
                return decode(tokens, *args, **kwargs)
            model.decode = record_decode
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            criterion.register_forward_pre_hook(
                lambda module, args: labels_seen.append(args[1].detach().clone()))
            loss = train_batch(model, source, target,
                               torch.optim.SGD(model.parameters(), lr=0.01),
                               criterion, torch.amp.GradScaler(enabled=False), False)
            assert len(decoder_inputs) == 1 and decoder_inputs[0].tolist() == [[2, 8, 3], [2, 6, 7]], \
                "Vstup dekodéra má začínať BOS a neobsahovať posledný token target."
            assert len(labels_seen) == 1 and labels_seen[0].tolist() == [8, 3, 0, 6, 7, 3], \
                "Správne odpovede majú byť posunuté o jeden token a obsahovať EOS, bez BOS."
            assert isinstance(loss, (int, float)) and math.isfinite(loss), \
                "train_batch má vrátiť konečné číslo loss."
            assert any(not torch.equal(old, new) for old, new in
                       zip(before, model.parameters())), "Váhy sa po kroku učenia nezmenili."

    def checkpoint_check():
        model = nn.Linear(2, 2)
        score, weights, epoch = select_checkpoint(model, {1: 10., 2: 30.}, 1, -1., None, None)
        assert score == 20. and epoch == 1 and weights is not None, \
            "Vyberaj podľa priemeru oboch validačných skóre."
        saved = {k: v.clone() for k, v in weights.items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        assert all(torch.equal(saved[k], weights[k]) for k in saved), \
            "Uložené váhy sa menia spolu s modelom. Vytvor ich nezávislú kópiu."
        lower = select_checkpoint(model, {1: 19., 2: 19.}, 2, score, weights, epoch)
        tied = select_checkpoint(model, {1: 20., 2: 20.}, 3, score, weights, epoch)
        assert lower[0] == tied[0] == score and lower[1] is weights and tied[1] is weights \
            and lower[2] == tied[2] == epoch, "Horší alebo zhodný výsledok nemá prepísať checkpoint."

    completed = 0
    for name, check in (("1/3 Príklady pre oba smery", examples_check),
                        ("2/3 Krok učenia", batch_check),
                        ("3/3 Validačný checkpoint", checkpoint_check)):
        try:
            check()
        except Exception as error:
            print(f"[DOPLŇ] {name}: {error}", flush=True)
        else:
            completed += 1
            print(f"[OK] {name}", flush=True)
    print(f"Dokončené: {completed}/3. Po kontrolách porovnaj model na validácii.", flush=True)
    return completed == 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Over tri doplnené úlohy na CPU bez tréningu")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("submission_neural.csv"))
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--vocab", type=int, default=4000)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bs", type=int, default=96)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--dev", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)
    if min(args.max_epochs, args.bs, args.eval_every, args.heads, args.layers) < 1:
        parser.error("Počet epoch, veľkosť dávky, interval hodnotenia a rozmery musia byť kladné")
    if args.d < 1 or args.d % args.heads or args.max_len < 4 or args.minutes <= 0:
        parser.error("--d musí byť deliteľné --heads, --max_len >= 4 a --minutes > 0")

    if not check_exercises():
        parser.exit(1, "Najprv doplň tri úlohy v train.py. Slovníkový starter.py môžeš spustiť hneď.\n")
    if args.check:
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    pairs = load_pairs(args.data / "train.csv")
    with (args.data / "test.csv").open(encoding="utf-8", newline="") as f:
        test = list(csv.DictReader(f))
    training, validation = split_pairs(pairs, args.dev, args.seed)
    tokenizer = make_tokenizer(training, args.vocab)
    pad_id, bos, eos = (tokenizer.token_to_id(t) for t in ("<pad>", "<s>", "</s>"))
    tags = {1: tokenizer.token_to_id("<2sk>"), 2: tokenizer.token_to_id("<2zak>")}
    vocab = tokenizer.get_vocab_size()

    def encode(sentence):
        return tokenizer.encode(sentence).ids[:args.max_len - 2]

    examples = make_examples(training, encode, tags)

    def pad(sequences):
        length = max(map(len, sequences))
        return torch.tensor([s + [pad_id] * (length - len(s)) for s in sequences], device=device)

    model = Seq2Seq(vocab, pad_id, args.d, args.heads, args.layers,
                   args.dropout, args.max_len).to(device)

    @torch.no_grad()
    def translate(sources):
        predictions = []
        for start in range(0, len(sources), args.bs):
            source = pad(sources[start:start + args.bs])
            memory = model.encode(source)
            target = torch.full((len(source), 1), bos, device=device)
            for _ in range(args.max_len):
                next_token = model.decode(target, memory, source == pad_id)[:, -1].argmax(-1)
                target = torch.cat([target, next_token[:, None]], dim=1)
                if (target == eos).any(1).all():
                    break
            for tokens in target[:, 1:].tolist():
                tokens = tokens[:tokens.index(eos)] if eos in tokens else tokens
                predictions.append(tokenizer.decode(tokens).strip())
        return predictions

    def evaluate():
        scores, rows = {}, []
        for direction in (1, 2):
            sources = [z if direction == 1 else s for z, s in validation]
            references = [s if direction == 1 else z for z, s in validation]
            predictions = translate([[tags[direction]] + encode(s) for s in sources])
            scores[direction] = CHRF().corpus_score(predictions, [references]).score
            rows.extend((direction, i, source, reference, prediction)
                        for i, (source, reference, prediction) in
                        enumerate(zip(sources, references, predictions)))
        return scores, rows

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.98), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    total_steps = math.ceil(len(examples) / args.bs) * args.max_epochs
    warmup = max(1, min(1000, total_steps // 10))
    started, step = time.time(), 0
    best_score, best_weights, best_epoch = -1.0, None, None
    print(f"device={device}, train_pairs={len(training)}, validation_pairs={len(validation)}, "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        examples.sort(key=lambda pair: len(pair[0]) + random.random() * 4)
        batches = [examples[i:i + args.bs] for i in range(0, len(examples), args.bs)]
        random.shuffle(batches)
        loss_sum, token_count = 0.0, 0
        for batch in batches:
            source = pad([s for s, _ in batch])
            target = pad([[bos] + t + [eos] for _, t in batch])
            lr = args.lr * min((step + 1) / warmup,
                              0.5 * (1 + math.cos(math.pi * step / total_steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = train_batch(model, source, target, optimizer, criterion, scaler, use_amp)
            count = (target[:, 1:] != pad_id).sum().item()
            loss_sum += loss * count
            token_count += count
            step += 1
        elapsed = time.time() - started
        stop = elapsed >= args.minutes * 60 or epoch == args.max_epochs
        message = f"epoch={epoch}, loss={loss_sum / token_count:.3f}, minutes={elapsed / 60:.1f}"
        if epoch % args.eval_every == 0 or stop:
            model.eval()
            scores, _ = evaluate()
            message += f", zak→sk chrF={scores[1]:.2f}, sk→zak chrF={scores[2]:.2f}"
            best_score, best_weights, best_epoch = select_checkpoint(
                model, scores, epoch, best_score, best_weights, best_epoch)
        print(message, flush=True)
        if stop:
            break

    model.load_state_dict(best_weights)
    print(f"Vybraný checkpoint: epocha {best_epoch} z {epoch} dokončených epoch", flush=True)
    model.eval()
    scores, validation_rows = evaluate()
    predictions = translate([[tags[int(row["subtaskID"])]] + encode(row["input"]) for row in test])
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["subtaskID", "datapointID", "answer"])
        writer.writerows((row["subtaskID"], row["datapointID"], prediction)
                         for row, prediction in zip(test, predictions))
    with args.out.with_suffix(".dev.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["subtaskID", "input", "reference", "answer"])
        writer.writerows((direction, source, reference, prediction)
                         for direction, _, source, reference, prediction in validation_rows)
    metrics = {"seed": args.seed, "fit_pairs": len(training), "dev_pairs": len(validation),
               "epochs_completed": epoch, "best_epoch": best_epoch,
               "zak_to_sk_chrF": scores[1], "sk_to_zak_chrF": scores[2],
               "minutes": (time.time() - started) / 60,
               "config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}}
    args.out.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Hotovo: {args.out}, zak→sk chrF={scores[1]:.2f}, sk→zak chrF={scores[2]:.2f}", flush=True)
    for direction, _, source, reference, prediction in validation_rows[:3]:
        print(f"S{direction} vstup: {source}\n  správne: {reference}\n  model:   {prediction}")


if __name__ == "__main__":
    main()
