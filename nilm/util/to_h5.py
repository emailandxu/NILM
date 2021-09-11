from nilmtk.dataset_converters import convert_redd
from pathlib import Path
# import argparse  # to do add argparse


def to_h5(source=None, dest=None):
    if not source:
        lowfreq_root = Path("../dataset").joinpath("low_freq/")
        source_dir: Path = lowfreq_root.absolute()
    else:
        source_dir = Path(source)
    if not dest:
        dest_h5: Path = lowfreq_root.joinpath("redd_low.h5").absolute()
    else:
        dest_h5: Path = Path(dest)
    if not dest_h5.exists():
        print(f"Converting {source_dir}->{dest_h5}")
        convert_redd(source_dir.as_posix(), dest_h5.as_posix())
    else:
        print(f"Abort, the output h5 file already exists!")


if __name__ == "__main__":
    to_h5()
