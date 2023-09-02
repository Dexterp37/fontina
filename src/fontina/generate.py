import argparse
import random
import rstr

from pathlib import Path, PurePath
from typing import Optional

from trdg.generators import GeneratorFromRandom, GeneratorFromStrings

from fontina.config import load_config


class GeneratorFromRegex(GeneratorFromStrings):
    def __init__(
        self, regex_template: str, count: int, random_character_spacing: bool, **kwargs
    ):
        super().__init__(
            count=count,
            strings=self._random_strings(
                count,
                regex_template,
            ),
            **kwargs,
        )
        self.random_character_spacing = random_character_spacing

    def _random_strings(self, num_to_generate: int, regex_template: str):
        return [rstr.xeger(regex_template) for _ in range(0, num_to_generate)]

    def _clamp(self, x, range_min, range_max):
        return max(range_min, min(x, range_max))

    def __next__(self):
        if self.random_character_spacing:
            raw_spacing = random.gauss(10, 40)
            self.character_spacing = round(self._clamp(raw_spacing, 0, 50))
        return self.next()


def generate_font_data(
    output_dir: PurePath,
    num_samples: int,
    font_path: PurePath,
    random_character_spacing: bool,
    backgrounds_path: Optional[PurePath],
    regex_template: Optional[str],
):
    # TODO: we have to generate the samples from the same strings.
    generator = (
        GeneratorFromRegex(
            count=num_samples,
            regex_template=regex_template,
            random_character_spacing=random_character_spacing,
            fonts=[str(font_path)],
            # This size maps to the height of the images.
            size=105,
            # Background == 1 means plain white and is the default
            # if no background images are provided.
            background_type=1 if not backgrounds_path else None,
            image_dir=backgrounds_path if backgrounds_path else None,
            fit=True,
        )
        if regex_template
        else GeneratorFromRandom(
            # Length works a bit oddly as it doesn't seem to map to
            # the length of the desired string.
            length=2,
            fonts=[str(font_path)],
            # This size maps to the height of the images.
            size=105,
            # Background == 1 means plain white and is the default
            # if no background images are provided.
            background_type=1 if not backgrounds_path else None,
            image_dir=backgrounds_path if backgrounds_path else None,
            fit=True,
        )
    )

    # Store the data in a subdirectory named as the font file.
    font_dir_name = font_path.stem
    output_dir = PurePath(output_dir, font_dir_name)

    # Make sure the output dir exists.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels_file = PurePath(output_dir, "labels.csv")
    with open(labels_file, "w", encoding="utf8") as csv:
        csv.write("filename,words\n")

        for i, (img, label) in enumerate(generator):
            filename = f"{i}.png"
            img.save(str(PurePath(output_dir, filename)))
            csv.write(f"{filename},{label}\n")

            if i == num_samples:
                break


def get_parser():
    parser = argparse.ArgumentParser(description="Fontina dataset generator")
    parser.add_argument(
        "-c",
        "--config",
        metavar="CONFIG-FILE",
        help="path to the configuration file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="out-data",
        metavar="OUTPUT-DIR",
        help="path to the directory that will contain the outputs",
    )
    return parser


def main():
    args = get_parser().parse_args()

    config = load_config(args.config)

    fonts_config = config["fonts"]
    for f in fonts_config["classes"]:
        print(f"Generating font data for {f['path']}")
        generate_font_data(
            output_dir=PurePath(args.outdir),
            num_samples=fonts_config["samples_per_font"],
            font_path=PurePath(f["path"]),
            backgrounds_path=PurePath(fonts_config["backgrounds_path"])
            if "backgrounds_path" in fonts_config
            else None,
            random_character_spacing=fonts_config["random_character_spacing"],
            regex_template=fonts_config["regex_template"],
        )


if __name__ == "__main__":
    main()
