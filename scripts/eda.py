"""Análise exploratória do metadata.csv: distribuição de nódulos por imagem e tamanho das caixas."""

import argparse

from node21det.data.metadata import image_table, load_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    args = p.parse_args()

    meta = load_metadata(args.metadata)
    images = image_table(meta)
    print(f"imagens: {len(images)} | positivas: {int(images['positive'].sum())} | nódulos: {int(images['n_nodules'].sum())}")
    print("\nnódulos por imagem positiva:")
    print(images.loc[images["positive"], "n_nodules"].value_counts().sort_index().to_string())

    pos = meta[meta["label"] == 1]
    side = (pos["width"] * pos["height"]) ** 0.5
    print("\nlado equivalente da caixa (raiz da área, px na imagem de 1024):")
    print(side.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_string())
    print("\nprefixo do nome do arquivo (origem):")
    print(images["img_name"].str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
