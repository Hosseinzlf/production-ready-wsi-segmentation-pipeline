import os
import subprocess


def main() -> None:
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

    cmd = [
        "prefect",
        "deployment",
        "run",
        "wsi-segmentation-flow/wsi-segmentation",
        "--param",
        "wsi_path=data/KLMP45690052_001.svs",
        "--param",
        "output_path=outputs/mask.tiff",
        "--param",
        "config=config/config.yaml",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    print("Return code:", result.returncode)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)


if __name__ == "__main__":
    main()
