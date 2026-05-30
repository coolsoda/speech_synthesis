{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c10fc77-2219-4b10-a106-6ea41bb3de28",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "Baseline augmentation methods for L-Flow.\n",
    "\n",
    "This package contains:\n",
    "- speed perturbation;\n",
    "- VTLP;\n",
    "- wrapper interfaces for external TTS/pathology-TTS baselines.\n",
    "\"\"\"\n",
    "\n",
    "from .speed_perturb import run_speed_perturbation\n",
    "from .vtlp import run_vtlp\n",
    "from .tts_baselines import (\n",
    "    run_external_tts_baseline,\n",
    "    convert_completed_manifest_to_metadata,\n",
    ")\n",
    "\n",
    "__all__ = [\n",
    "    \"run_speed_perturbation\",\n",
    "    \"run_vtlp\",\n",
    "    \"run_external_tts_baseline\",\n",
    "    \"convert_completed_manifest_to_metadata\",\n",
    "]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
