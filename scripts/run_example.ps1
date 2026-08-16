param([ValidateSet('train','validate','analyze')][string]$Command = 'train')
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
python scripts/create_example_images.py
python -m gray.cli $Command --config source/configs/example_classification/example_classification.yaml
