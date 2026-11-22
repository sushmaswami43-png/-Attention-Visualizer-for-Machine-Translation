"""Download the Multi30k task-1 parallel corpus (German-English, ~29k training pairs)."""
import gzip
import shutil
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
FILES = [f"{split}.{lang}" for split in ("train", "val", "test_2016_flickr") for lang in ("de", "en")]


def main(out_dir: Path = Path("data")) -> None:
    out_dir.mkdir(exist_ok=True)
    for name in FILES:
        target = out_dir / name
        if target.exists():
            continue
        archive = out_dir / f"{name}.gz"
        urllib.request.urlretrieve(f"{BASE}/{name}.gz", archive)
        with gzip.open(archive, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        archive.unlink()
        print("downloaded", name)


if __name__ == "__main__":
    main()
