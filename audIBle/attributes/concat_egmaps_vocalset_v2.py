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

def parse_filename(filename):
    """
    Parse VocalSet OpenSmile feature filenames.
    Adds 'raw_name' = canonical audio file ID (short version if available, otherwise full filename).
    """
    name_without_ext = filename[:-4]
    parts = name_without_ext.split('_')

    singer_abbreviations = ['m3', 'm5', 'm10', 'f2', 'f8']

    audio_start_idx = None
    for i, part in enumerate(parts):
        if part in singer_abbreviations:
            audio_start_idx = i
            break

    if audio_start_idx is not None:
        prefix_parts = parts[:audio_start_idx]
        audio_parts = parts[audio_start_idx:]

        full_singer = prefix_parts[0]
        exercise = prefix_parts[1]
        if exercise == "long":
            exercise = "long_tones"

        technique_parts = prefix_parts[2:]
        technique = "_".join(technique_parts) if technique_parts else "unknown"

        vowel = audio_parts[-1]
        audio_filename = "_".join(audio_parts)

        raw_name = audio_filename  # short version is cleanest

    else:
        full_singer = parts[0]
        exercise = parts[1]
        if exercise == "long":
            exercise = "long_tones"

        technique_parts = parts[2:-1]
        technique = "_".join(technique_parts) if technique_parts else "unknown"

        vowel = parts[-1]
        audio_filename = "_".join(parts[1:])  # skip singer

        raw_name = name_without_ext  # fallback to full filename

    return {
        "singer": full_singer,
        "exercise_type": exercise,
        "vocal_technique": technique,
        "vowel": vowel,
        "audio_filename": audio_filename,
        "raw_name": raw_name,  # <-- new field
    }


# def parse_filename(filename):
#     """
#     Parse filename to extract singer, exercise, technique, and vowel.
    
#     Examples:
#     - male3_scales_fast_piano_m3_scales_f_fast_piano_a.csv -> m3_scales_f_fast_piano_a
#     - male3_scales_lip_trill_m3_scales_lip_trill_u.csv -> m3_scales_lip_trill_u
    
#     The pattern is: [prefix]_[audio_filename]
#     Where audio_filename follows: singer_exercise_technique_vowel
#     """
#     # Remove .csv extension
#     name_without_ext = filename[:-4]
#     parts = name_without_ext.split('_')
    
#     if len(parts) < 4:
#         return {
#             'singer': 'unknown',
#             'exercise_type': 'unknown',
#             'vocal_technique': 'unknown',
#             'vowel': 'unknown',
#             'audio_filename': name_without_ext
#         }
    
#     # Find the audio filename part by looking for singer abbreviations (m3, f2, etc.)
#     audio_start_idx = None
    
#     # Look for abbreviated singer names (m3, m5, m10, f2, f8)
#     singer_abbreviations = ['m3', 'm5', 'm10', 'f2', 'f8']
    
#     for i, part in enumerate(parts):
#         if part in singer_abbreviations:
#             # Check if this looks like the start of an audio filename
#             remaining_parts = parts[i:]
#             if len(remaining_parts) >= 4:  # Need at least singer_exercise_technique_vowel
#                 audio_start_idx = i
#                 break
    
#     if audio_start_idx is None:
#         # Fallback: assume audio filename is in the last parts
#         audio_start_idx = max(0, len(parts) - 5)
    
#     audio_parts = parts[audio_start_idx:]
    
#     if len(audio_parts) < 4:
#         # Not enough parts, return what we can
#         return {
#             'singer': parts[0] if parts else 'unknown',
#             'exercise_type': 'unknown',
#             'vocal_technique': 'unknown', 
#             'vowel': audio_parts[-1] if audio_parts else 'unknown',
#             'audio_filename': '_'.join(audio_parts) if audio_parts else name_without_ext
#         }
    
#     # Parse audio filename: singer_exercise_technique(s)_vowel
#     audio_singer = audio_parts[0]  # e.g., m3, f2
#     exercise = audio_parts[1]      # e.g., scales, arpeggios, long
#     vowel = audio_parts[-1]        # Always the last part (a, e, i, o, u, etc.)
    
#     # Everything between exercise and vowel is the technique
#     technique_parts = audio_parts[2:-1]
    
#     # Clean up the technique - remove any singer abbreviations or exercise types that got mixed in
#     cleaned_technique_parts = []
#     for part in technique_parts:
#         if (part not in singer_abbreviations and 
#             part not in ['scales', 'arpeggios', 'long', 'excerpts'] and
#             part not in ['a', 'e', 'i', 'o', 'u']):  # Not a vowel either
#             cleaned_technique_parts.append(part)
    
#     technique = '_'.join(cleaned_technique_parts) if cleaned_technique_parts else 'unknown'
    
#     # Handle special cases
#     if exercise == 'long':
#         exercise = 'long_tones'
    
#     # Map singer abbreviations to full names
#     singer_mapping = {
#         'm3': 'male3', 'm5': 'male5', 'm10': 'male10',
#         'f2': 'female2', 'f8': 'female8'
#     }
#     full_singer = singer_mapping.get(audio_singer, audio_singer)
    
#     result = {
#         'singer': full_singer,
#         'exercise_type': exercise,
#         'vocal_technique': technique,
#         'vowel': vowel,
#         'audio_filename': '_'.join(audio_parts)
#     }
    
#     return result

glob_df = []
processing_errors = []
wavpath = os.path.join(os.environ["DATA_ROOT"],"VocalSet/FULL",args.split)
feat_path = os.path.join(os.environ["DATA_ROOT"],"VocalSet/features", args.split)

for f in glob.glob(f"{feat_path}/*.csv"):
    # Extract feature filename without path
    filename = f.split('/')[-1]
    bname = filename[:-4]  # Remove .csv extension
    
    print(f"Processing: {filename}")
    
    # Parse the filename to extract metadata
    parsed = parse_filename(filename)
    
    # Read the CSV file (skip OpenSMILE header comments with @)
    try:
        data = pd.read_csv(f, engine='python', comment='@', names=header[1:-4])  # Exclude name and metadata columns
        data = data.drop_duplicates()
        
        # Reconstruct original audio path
        audio_path = os.path.join(wavpath, parsed['audio_filename'].replace('_', '/') + '.wav')
        
        # Add metadata columns
        data['name'] = bname
        data['audio_path'] = audio_path
        data['singer'] = parsed['singer']
        data['exercise_type'] = parsed['exercise_type']
        data['vocal_technique'] = parsed['vocal_technique']
        data['vowel'] = parsed['vowel']
        data['audio_id'] = parsed['raw_name']
        
        glob_df.append(data)
        print(f"  SUCCESS: Added {len(data)} rows - Singer: {parsed['singer']}, Exercise: {parsed['exercise_type']}, Technique: {parsed['vocal_technique']}, Vowel: {parsed['vowel']}")
        
    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")
        processing_errors.append((filename, str(e)))
        continue

# Print summary
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Files processed successfully: {len(glob_df)}")
print(f"Files with errors: {len(processing_errors)}")

if processing_errors:
    print(f"\nFiles with processing errors:")
    for filename, error in processing_errors[:10]:  # Show first 10
        print(f"  - {filename}: {error}")
    if len(processing_errors) > 10:
        print(f"  ... and {len(processing_errors) - 10} more")

# Concatenate all dataframes
if glob_df:
    glob_df = pd.concat(glob_df, ignore_index=True)
    print(f"\nFinal dataset:")
    print(f"Total samples: {len(glob_df)}")
    print(f"Singers: {sorted(glob_df['singer'].unique())}")
    print(f"Exercise types: {sorted(glob_df['exercise_type'].unique())}")
    print(f"Vocal techniques: {sorted(glob_df['vocal_technique'].unique())}")
    print(f"Vowels: {sorted(glob_df['vowel'].unique())}")
    
    # Save to CSV
    output_path = os.path.join(os.environ["DATA_ROOT"],"VocalSet/features/concat",args.split)
    os.makedirs(output_path,exist_ok=True)
    output_file = os.path.join(output_path,"features_egemaps_vocalset.csv")
    glob_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")
else:
    print("No data processed successfully.")