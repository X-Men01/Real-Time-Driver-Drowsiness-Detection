
## Dataset Versions

### 1. Raw Dataset (Before Cleaning)
- Located in: `raw/Dataset_Before_Cleaning/`
- Contains original, unprocessed images
- Known issues identified through automated analysis:
  - Blurry images (2,656 instances)
  - Grayscale images (2,000 instances)
  - Odd aspect ratio images (1,937 instances)
  - Near-duplicate images (1,378 instances)
  - Odd-sized images (857 instances)
  - Dark images (563 instances)
  - Exact duplicates (530 instances)
  - Low information images (9 instances)
  - Light images (2 instances)

### 2. Auto-Cleaned Dataset
- Located in: `raw/Dataset_After_Auto_Cleaning/`
- Automatically processed using CleanVision library
- Cleaning process removed:
  - Duplicate images (both exact and near duplicates)
  - Blurry images
  - Images with poor lighting (too dark/light)
  - Images with odd aspect ratios
  - Low information content images
- Remaining issues after auto-cleaning:
  - Grayscale images (1,760 instances)
  - Odd-sized images (366 instances)

### 3. Manually Cleaned Dataset (Final Version)
- Located in: `processed/Dataset_After_Manual_Cleaning/`
- Manually verified and cleaned dataset
- Highest quality subset of images
- Recommended for training and testing models

## Categories Description

1. **Eye States**
   - `Open_Eye`: Images of subjects with eyes clearly open
   - `Close_Eye`: Images of subjects with eyes closed

2. **Mouth States**
   - `Yawn`: Images of subjects yawning
   - `No_Yawn`: Images of subjects with mouth closed or in neutral position



# Dataset Analysis Report

## 1. Dataset Statistics

| Category   | Before Cleaning | After Auto Cleaning | After Manual Cleaning |
|------------|-----------------|---------------------|-----------------------|
| Close_Eye  | 14569           | 9122                | 7183                  |
| No_Yawn    | 10816           | 8241                | 8174                  |
| Open_Eye   | 10474           | 7038                | 6546                  |
| Yawn       | 10195           | 7443                | 6091                  |
| **Total**  | **46054**       | **31844**           | **27994**             |

## 2. Changes Between Stages

- **Before Cleaning → After Auto Cleaning**: -30.9% change  
- **After Auto Cleaning → After Manual Cleaning**: -12.1% change
![Dataset_stat](https://github.com/user-attachments/assets/7f259222-2096-4f60-8eef-1bba30d11c15)

