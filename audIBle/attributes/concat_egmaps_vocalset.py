import pandas as pd
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split",type=str,choices=["train","test"])
args = parser.parse_args()

header = [
    "name",
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_percentile50.0",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_stddevRisingSlope",
    "loudness_sma3_meanFallingSlope",
    "loudness_sma3_stddevFallingSlope",
    "spectralFlux_sma3_amean",
    "spectralFlux_sma3_stddevNorm",
    "mfcc1_sma3_amean",
    "mfcc1_sma3_stddevNorm",
    "mfcc2_sma3_amean",
    "mfcc2_sma3_stddevNorm",
    "mfcc3_sma3_amean",
    "mfcc3_sma3_stddevNorm",
    "mfcc4_sma3_amean",
    "mfcc4_sma3_stddevNorm",
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "shimmerLocaldB_sma3nz_stddevNorm",
    "HNRdBACF_sma3nz_amean",
    "HNRdBACF_sma3nz_stddevNorm",
    "logRelF0-H1-H2_sma3nz_amean",
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    "logRelF0-H1-A3_sma3nz_amean",
    "logRelF0-H1-A3_sma3nz_stddevNorm",
    "F1frequency_sma3nz_amean",
    "F1frequency_sma3nz_stddevNorm",
    "F1bandwidth_sma3nz_amean",
    "F1bandwidth_sma3nz_stddevNorm",
    "F1amplitudeLogRelF0_sma3nz_amean",
    "F1amplitudeLogRelF0_sma3nz_stddevNorm",
    "F2frequency_sma3nz_amean",
    "F2frequency_sma3nz_stddevNorm",
    "F2bandwidth_sma3nz_amean",
    "F2bandwidth_sma3nz_stddevNorm",
    "F2amplitudeLogRelF0_sma3nz_amean",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm",
    "F3frequency_sma3nz_amean",
    "F3frequency_sma3nz_stddevNorm",
    "F3bandwidth_sma3nz_amean",
    "F3bandwidth_sma3nz_stddevNorm",
    "F3amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "alphaRatioV_sma3nz_amean",
    "alphaRatioV_sma3nz_stddevNorm",
    "hammarbergIndexV_sma3nz_amean",
    "hammarbergIndexV_sma3nz_stddevNorm",
    "slopeV0-500_sma3nz_amean",
    "slopeV0-500_sma3nz_stddevNorm",
    "slopeV500-1500_sma3nz_amean",
    "slopeV500-1500_sma3nz_stddevNorm",
    "spectralFluxV_sma3nz_amean",
    "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean",
    "mfcc1V_sma3nz_stddevNorm",
    "mfcc2V_sma3nz_amean",
    "mfcc2V_sma3nz_stddevNorm",
    "mfcc3V_sma3nz_amean",
    "mfcc3V_sma3nz_stddevNorm",
    "mfcc4V_sma3nz_amean",
    "mfcc4V_sma3nz_stddevNorm",
    "alphaRatioUV_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "spectralFluxUV_sma3nz_amean",
    "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
    "StddevUnvoicedSegmentLength",
    "equivalentSoundLevel_dBp",
    "audio_path",
    "singer",
    "exercise_type", 
    "vocal_technique",
    "vowel"
]

glob_df = []
wavpath = os.path.join(os.environ["DATA_ROOT"],"VocalSet/FULL",args.split)
feat_path = os.path.join(os.environ["DATA_ROOT"],"VocalSet/features", args.split)
for f in glob.glob(f"{feat_path}/*.csv"):
    # Extract feature filename without path and extension
    bname = f.split('/')[-1][:-4]
    print(f"Processing: {bname}")
    
    # Read the CSV file (skip OpenSMILE header comments with @)
    try:
        data = pd.read_csv(f, engine='python', comment='@', names=header[1:-4])  # Exclude name and metadata columns
        data = data.drop_duplicates()
        
        # Reconstruct original audio path by converting underscores back to slashes
        audio_path = os.path.join(wavpath, bname.replace('_', '/') + '.wav')
        
        # Extract metadata from the path structure
        path_parts = bname.split('_')
        
        # Parse the path structure: singer_exercise_technique_[key_]details_vowel
        singer = path_parts[0]  # e.g., 'female6', 'female7'
        exercise_type = path_parts[1]  # e.g., 'arpeggios', 'scales', 'long', 'excerpts'
        
        # Handle different path structures
        if exercise_type == 'long':
            exercise_type = 'long_tones'
            vocal_technique = path_parts[2]
            vowel = path_parts[-1] if len(path_parts) > 3 else 'unknown'
        elif len(path_parts) >= 4:
            # Handle cases like arpeggios_c_fast_forte or arpeggios_belt_1
            if path_parts[2] in ['c', 'f']:  # Key indicators
                vocal_technique = '_'.join(path_parts[2:-1])
                vowel = path_parts[-1]
            else:
                vocal_technique = path_parts[2]
                vowel = path_parts[-1]
        else:
            vocal_technique = 'unknown'
            vowel = path_parts[-1] if len(path_parts) > 2 else 'unknown'
        
        # Add metadata columns
        data['name'] = bname
        data['audio_path'] = audio_path
        data['singer'] = singer
        data['exercise_type'] = exercise_type
        data['vocal_technique'] = vocal_technique
        data['vowel'] = vowel
        
        glob_df.append(data)
        
    except Exception as e:
        print(f"Error processing {f}: {e}")
        continue

# Concatenate all dataframes
if glob_df:
    glob_df = pd.concat(glob_df, ignore_index=True)
    print(f"\nTotal samples: {len(glob_df)}")
    print(f"Singers: {glob_df['singer'].unique()}")
    print(f"Exercise types: {glob_df['exercise_type'].unique()}")
    print(f"Vocal techniques: {glob_df['vocal_technique'].unique()}")
    print(f"Vowels: {glob_df['vowel'].unique()}")
    
    # Save to CSV
    output_path = os.path.join(os.environ["DATA_ROOT"],"VocalSet/features/concat",args.split)
    os.makedirs(output_path,exist_ok=True)
    output_file = os.path.join(output_path,"features_egemaps_vocalset.csv")
    glob_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")
else:
    print("No data processed successfully.")