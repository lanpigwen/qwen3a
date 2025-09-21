import argparse

def load_lines(p):
    with open(p,'r',encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--ref_file", required=True)
    ap.add_argument("--ignore_sil", type=str, default="true")
    args = ap.parse_args()
    ignore_sil = args.ignore_sil.lower() in ["1","true","yes","y"]
    preds = load_lines(args.pred_file)
    refs = load_lines(args.ref_file)
    assert len(preds)==len(refs), "预测与参考行数不一致"
    total_tok = 0
    correct = 0
    for pr, rf in zip(preds, refs):
        p_toks = pr.split()
        r_toks = rf.split()
        L = min(len(p_toks), len(r_toks))
        for i in range(L):
            pt, rt = p_toks[i], r_toks[i]
            if ignore_sil and (pt=="sil" or rt=="sil"):
                continue
            total_tok += 1
            if pt == rt:
                correct += 1
    acc = correct / total_tok if total_tok>0 else 0.0
    print(f"Tokens(有效): {total_tok}  Correct: {correct}  Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
