# Dataset Documentation: Infant Resting State EEG

## Overview

This dataset contains resting state EEG recordings from 103 normal infants during their first year of life. The data follows the Brain Imaging Data Structure (BIDS) version 1.6.0 format.

**Institution**: Neurodevelopment Research Unit, Instituto de Neurobiología, Universidad Nacional Autónoma de México

**Version**: 1.0.1 (May 25th, 2023)

**DOI**: doi:10.18112/openneuro.ds004577.v1.0.1

**License**: CC0

## Subject Demographics

- **Total Subjects**: 103 infants
- **Female**: 41
- **Male**: 62
- **Age Range**: First year of life

## Data Distribution

- **Total Recordings**: 130 EEG sessions
- **Subjects with 1 recording**: 81
- **Subjects with 2 recordings**: 18
- **Subjects with 3 recordings**: 3
- **Subjects with 4 recordings**: 1
- **Sessions**: Distributed across 4 sessions

## File Structure

### Root Level Files

#### `participants.tsv`
Tab-separated file containing basic demographic information for all subjects.

**Columns**:
- `participant_id`: Subject identifier (sub-NORB00001 to sub-NORB00103)
- `sex`: Biological sex (M/F)

#### `dataset_description.json`
JSON file containing dataset metadata including authors, funding information, ethics approvals, and references to related publications.

#### `README`
Plain text file with dataset overview and recording distribution summary.

#### `CHANGES`
Version history and changelog for the dataset.

### Subject Organization

Each subject is organized under `sub-NORB#####/` directories with session subdirectories `ses-#/`.

### Session Level Files

#### `sub-NORB#####_ses-#_scans.tsv`
Records metadata about the scanning session.

**Columns**:
- `filename`: Path to the EEG data file
- `age_acq_time`: Age at acquisition in years (decimal format)

### EEG Data Files

Located in `sub-NORB#####/ses-#/eeg/` directories:

#### `sub-NORB#####_ses-#_task-EEG_eeg.edf`
Raw EEG data in European Data Format (EDF).

**Technical Specifications**:
- **Sampling Frequency**: 200 Hz
- **Recording Type**: Continuous
- **Power Line Frequency**: 60 Hz
- **EEG Placement Scheme**: 10-10 system
- **Reference**: Common reference
- **Duration**: Variable (approximately 528-1784 seconds)

#### `sub-NORB#####_ses-#_task-EEG_eeg.json`
Sidecar JSON file containing recording parameters and metadata.

**Key Fields**:
- `TaskName`: Task identifier (EEG)
- `TaskDescription`: Description (Resting EEG)
- `SamplingFrequency`: Sampling rate in Hz
- `EEGChannelCount`: Number of EEG channels (typically 19-21)
- `EEGPlacementScheme`: Electrode placement system
- `EEGReference`: Reference type
- `RecordingDuration`: Length of recording in seconds

#### `sub-NORB#####_ses-#_task-EEG_channels.tsv`
Channel-specific information for each electrode.

**Columns**:
- `name`: Channel name (e.g., Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ)
- `type`: Channel type (EEG)
- `units`: Measurement units (uV)
- `description`: Channel description (electrode)
- `sampling_frequency`: Sampling rate in Hz (200)
- `low_cutoff`: High-pass filter cutoff in Hz (0.5)
- `high_cutoff`: Low-pass filter cutoff in Hz (30)
- `notch`: Notch filter specification (n/a)
- `status`: Channel quality status (good)

#### `sub-NORB#####_ses-#_electrodes.tsv`
Spatial coordinates and physical properties of electrodes.

**Columns**:
- `name`: Electrode name
- `x`: X-coordinate in mm
- `y`: Y-coordinate in mm
- `z`: Z-coordinate in mm
- `type`: Electrode type (cup)
- `material`: Electrode material (Ag/AgCl)

#### `sub-NORB#####_ses-#_coordsystem.json`
Coordinate system specification for electrode positions.

**Fields**:
- `EEGCoordinateSystem`: Coordinate system used (MNI305)
- `EEGCoordinateUnits`: Units of measurement (mm)

#### `sub-NORB#####_ses-#_task-EEG_events.tsv`
Event markers and annotations during recording.

**Columns**:
- `onset`: Event onset time in seconds
- `duration`: Event duration in seconds
- `trial_type`: Type of event (discontinuity, eyes_closed, eyes_open)
- `value`: Event code
- `sample`: Sample number at event onset

#### `eeg.zip`
Compressed archive containing additional EEG-related files.

#### `Ses#-P#-csv.csv`
Session-specific CSV file with additional data.

## Derivatives

Located in `derivatives/NeuronicEEG/` directory, containing processed annotations.

### Annotation Files

#### `sub-NORB#####_ses-#_task-EEG_annotations.tsv`
Visual inspection annotations for quantitative analysis.

**Columns**:
- `onset`: Annotation start time in seconds
- `duration`: Annotation duration in seconds (typically 2.56s)
- `label`: Annotation label (eyes_closed)

#### `sub-NORB#####_ses-#_task-EEG_annotations.json`
Metadata for annotation files including description and intended use.

## Dataset Overview Files

Located in `Dataset-Overview/` directory:

### `all_metadata.csv`
Consolidated metadata for all recordings across all subjects.

**Columns**:
- `subject_id`: Subject identifier
- `file_path`: Path to EEG file
- `duration_sec`: Recording duration in seconds
- `sfreq`: Sampling frequency in Hz
- `n_channels`: Number of channels
- `channel_names`: List of channel names
- `start_datetime`: Recording start timestamp

### `sub-NORB#####_metadata.json`
Individual subject metadata files containing session-level information structured in JSON format.

**Structure**:
- `subject_id`: Subject identifier
- `sessions`: Array of session objects containing:
  - `file_path`: Path to EEG data
  - `duration_sec`: Duration in seconds
  - `sfreq`: Sampling frequency
  - `n_channels`: Channel count
  - `channel_names`: Array of channel names
  - `start_datetime`: ISO 8601 timestamp

## Key References

1. Otero GA, Harmony T, et al. QEEG norms for the first year of life. Early Hum Dev, 87: 691-703, 2011.
2. Bosch-Bayard Jorge, et al. 3D Statistical Parametric Mapping of quiet sleep EEG in the first year of life. Neuroimage, 59: 3297-3308, 2012.
3. Bosch-Bayard J, et al. EEG effective connectivity during the first year of life mirrors brain synaptogenesis, myelination, and early right hemisphere predominance. NeuroImage 252 (2022) 119035.

## Data Acquisition

All recordings were conducted following ethical principles outlined in the Helsinki Declaration with approval from the Ethics Committee of the Instituto de Neurobiología, Universidad Nacional Autónoma de México.

## Technical Notes

- All EEG recordings use the 10-10 electrode placement system
- Recordings are referenced to a common reference
- Filters applied: 0.5 Hz high-pass, 30 Hz low-pass
- Data format complies with BIDS 1.6.0 specification
- Coordinate system: MNI305 for electrode positions
