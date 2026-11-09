import csv
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import tqdm

# Set up the YouTube Data API client
API_KEY = 'AIzaSyBo6mTrsODrJ2G2Zr5viRmUHfEk0h5J_Yw'
youtube = build('youtube', 'v3', developerKey=API_KEY)

# define function to check if video is available on YouTube
def is_valid_video(video):
    try:
        status = video["status"]
        content_details = video["contentDetails"]
        region_restriction = content_details.get("regionRestriction")
        default_audio_language = video["snippet"].get("defaultAudioLanguage")
        if status["uploadStatus"] == "processed" and status["privacyStatus"] == "public":
            if region_restriction and "US" not in region_restriction.get("allowed", []):
                return False
            if default_audio_language and default_audio_language != "en":
                return False
            return True
        else:
            return False
    except HttpError as error:
        print(f"An error occurred: {error}")
        return False


# define input and output files
input_file = 'data/avspeech_train.csv'
output_file = 'data/avspeech_train_cleaned.csv'

# read video IDs from input file
with open(input_file, "r") as csvfile:
    reader = csv.reader(csvfile)
    input_data = [row for row in reader]

# remove duplicates from video IDs list
video_ids = list(set([row[0] for row in input_data]))
print(len(video_ids))

# break video IDs into chunks of 50
chunks = [video_ids[i:i + 50] for i in range(0, len(video_ids), 50)]

# find valid videos
valid_videos = []
for chunk in tqdm.tqdm(chunks):
    video_ids_str = ",".join(chunk)
    video_info = youtube.videos().list(part="id,snippet,status,contentDetails", id=video_ids_str).execute()
    for video in video_info["items"]:
        if is_valid_video(video) and video["id"] in video_ids:
            valid_videos.append(video["id"])

# write valid videos to output file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for row in input_data:
        video_id = row[0]
        if video_id in valid_videos:
            writer.writerow(row)
