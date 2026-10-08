import re
from pathlib import Path

import numpy as np

import config


ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def parse_header(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.upper() == "ENVI" or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if value.startswith("{") and "}" not in value:
            parts = [value]
            while i < len(lines):
                parts.append(lines[i].strip())
                if "}" in lines[i]:
                    i += 1
                    break
                i += 1
            value = " ".join(parts)

        out[key] = value

    return out


def parse_number_list(value: str | None) -> np.ndarray:
    if not value:
        return np.array([], dtype=np.float32)

    cleaned = value.replace("{", " ").replace("}", " ")
    values = []

    for token in re.split(r"[\s,]+", cleaned.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            pass

    return np.asarray(values, dtype=np.float32)


def find_binary(hdr: Path) -> Path:
    base = hdr.with_suffix("")
    candidates = [
        base.with_suffix(".raw"),
        base.with_suffix(".RAW"),
        base.with_suffix(".dat"),
        base.with_suffix(".DAT"),
        base,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    raise FileNotFoundError(f"No ENVI binary file found for {hdr}")


def auto_find_headers(input_dir=None) -> tuple[Path, Path, Path]:
    if input_dir is None:
        input_dir = config.INPUT_DIR

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}"
        )

    headers = (
        sorted(input_dir.glob("*.hdr"))
        + sorted(input_dir.glob("*.HDR"))
    )

    if not headers:
        raise FileNotFoundError(
            f"No ENVI header files found in {input_dir}"
        )

    dark = [
        path
        for path in headers
        if "darkref" in path.name.lower()
    ]
    white = [
        path
        for path in headers
        if "whiteref" in path.name.lower()
    ]
    main = [
        path
        for path in headers
        if "darkref" not in path.name.lower()
        and "whiteref" not in path.name.lower()
    ]

    if (
        len(main) != 1
        or len(white) != 1
        or len(dark) != 1
    ):
        raise RuntimeError(
            "Could not uniquely identify main, white, and dark headers "
            f"in {input_dir}. Keep one matching acquisition in each "
            "dataset folder."
        )

    return main[0], white[0], dark[0]


class EnviCube:
    def __init__(self, hdr: Path):
        self.hdr = hdr
        self.header = parse_header(hdr)
        self.binary = find_binary(hdr)

        self.lines = int(float(self.header["lines"]))
        self.samples = int(float(self.header["samples"]))
        self.bands = int(float(self.header["bands"]))
        self.offset = int(float(self.header.get("header offset", "0")))
        self.byte_order = int(float(self.header.get("byte order", "0")))
        self.interleave = self.header.get("interleave", "bil").strip().lower()

        code = int(float(self.header["data type"]))
        dtype = np.dtype(ENVI_DTYPES[code])
        self.dtype = dtype.newbyteorder("<" if self.byte_order == 0 else ">")

        if self.interleave == "bil":
            shape = (self.lines, self.bands, self.samples)
        elif self.interleave == "bip":
            shape = (self.lines, self.samples, self.bands)
        elif self.interleave == "bsq":
            shape = (self.bands, self.lines, self.samples)
        else:
            raise ValueError(
                f"Unsupported ENVI interleave: {self.interleave}"
            )

        self.mm = np.memmap(
            self.binary,
            dtype=self.dtype,
            mode="r",
            offset=self.offset,
            shape=shape,
        )

        wavelengths = parse_number_list(self.header.get("wavelength"))
        if wavelengths.size == self.bands:
            self.wavelengths = wavelengths
        else:
            self.wavelengths = np.arange(
                self.bands,
                dtype=np.float32,
            )

    def yxb(self) -> np.ndarray:
        if self.interleave == "bil":
            return np.asarray(self.mm).transpose(0, 2, 1)
        if self.interleave == "bip":
            return np.asarray(self.mm)
        return np.asarray(self.mm).transpose(1, 2, 0)


def calibrate_reflectance(
    main_hdr: Path,
    white_hdr: Path,
    dark_hdr: Path,
):
    main = EnviCube(main_hdr)
    white = EnviCube(white_hdr)
    dark = EnviCube(dark_hdr)

    if white.samples != main.samples or white.bands != main.bands:
        raise ValueError(
            "White reference dimensions do not match the main cube."
        )

    if dark.samples != main.samples or dark.bands != main.bands:
        raise ValueError(
            "Dark reference dimensions do not match the main cube."
        )

    raw = main.yxb()
    white_raw = white.yxb().astype(np.float32)
    dark_raw = dark.yxb().astype(np.float32)

    white_mean = np.mean(white_raw, axis=0)
    dark_mean = np.mean(dark_raw, axis=0)

    denominator = white_mean - dark_mean
    invalid_reference = np.abs(denominator) < config.EPS
    raw_clipped = raw == config.RAW_CLIPPING_CODE

    reflectance = (
        raw.astype(np.float32) - dark_mean[None, :, :]
    ) / (denominator[None, :, :] + config.EPS)

    invalid = (
        raw_clipped
        | invalid_reference[None, :, :]
        | ~np.isfinite(reflectance)
    )

    reflectance[invalid] = np.nan

    return main, reflectance.astype(np.float32), invalid
