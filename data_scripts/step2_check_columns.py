import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "clean_data" / "normal"
TIME_COL = "ts"


def main():
    files = sorted(CLEAN_DIR.glob("*.csv"))
    if not files:
        print("No files in clean_data/")
        return

    ref = pd.read_csv(files[0], nrows=1)
    ref_cols = [c for c in ref.columns if c != TIME_COL]

    print("Reference file:", files[0].name)
    print("Num feature cols (D):", len(ref_cols))
    print()

    ok = True
    for f in files[1:]:
        df = pd.read_csv(f, nrows=1)
        cols = [c for c in df.columns if c != TIME_COL]

        if cols != ref_cols:
            ok = False
            print("[ERROR] Column mismatch:", f.name)
            ref_set, cur_set = set(ref_cols), set(cols)
            only_ref = sorted(ref_set - cur_set)
            only_cur = sorted(cur_set - ref_set)
            if only_ref:
                print("   missing vs ref:", only_ref)
            if only_cur:
                print("   extra vs ref:", only_cur)
            if ref_set == cur_set and cols != ref_cols:
                print("   same set but DIFFERENT ORDER")

    if ok:
        print("[OK] All cleaned files have identical feature columns and order.")


if __name__ == "__main__":
    main()
