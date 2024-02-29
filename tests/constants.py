PREPROCESSED_DATASET_DIR = 'preprocessed_dataset'
SUBSET = 'GrooveMIDI_processed_validation'

DEFAULT_FILTERS = {
    "drummer": None,  # ["drummer1", ..., and/or "session9"]
    "session": None,  # ["session1", "session2", and/or "session3"]
    "loop_id": None,
    "master_id": None,
    "style_primary": None,  # [funk, latin, jazz, rock, gospel, punk, hiphop, pop, soul, neworleans, afrobeat]
    "bpm": None,  # [(range_0_lower_bound, range_0_upper_bound), ..., (range_n_lower_bound, range_n_upper_bound)]
    "beat_type": ["beat"],  # ["beat" or "fill"]
    "time_signature": ["4-4"],  # ["4-4", "3-4", "6-8"]
    "full_midi_filename": None,  # list_of full_midi_filenames
    "full_audio_filename": None  # list_of full_audio_filename
}

ROCK_FILTERS = DEFAULT_FILTERS.copy()
ROCK_FILTERS["style_primary"] = ["rock"]

AFROBEAT_FILTERS = DEFAULT_FILTERS.copy()
AFROBEAT_FILTERS["style_primary"] = ["afrobeat"]