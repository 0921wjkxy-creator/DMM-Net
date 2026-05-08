# DMM-Net

This repository contains the implementation of **DMM-Net: A Dual-Memory-Driven Multimodal Sequence Fusion Network for Retrogressive Thaw Slump Mapping Based on Optical and SAR Images**.

## Overview

DMM-Net is designed for high-precision retrogressive thaw slump (RTS) mapping using optical and Synthetic Aperture Radar (SAR) remote sensing images. The framework integrates Sentinel-1A SAR and Sentinel-2A optical imagery and contains two core modules:

- Multimodal Memory Fusion Module (MMFM)
- Semantic Prototype Memory Module (SPMM)

## Dataset

The processed PNG dataset supporting this study is openly available on Zenodo:

https://doi.org/10.5281/zenodo.20079548

The Sentinel-1A SAR data and Sentinel-2A optical images used in this study are openly available from the European Space Agency (ESA) through the Copernicus Data Space Ecosystem:

https://dataspace.copernicus.eu

The Google Earth imagery used for visual interpretation was accessed through Google Earth:

https://earth.google.com

## Code Structure

```text
DMM-Net/
├── code/
│   ├── dataset/
│   ├── models/
│   ├── train_thermal_slide.py
│   ├── test_thermal_slide.py
│   └── utils.py
└── README.md
