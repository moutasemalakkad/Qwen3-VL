"""Simple test for fetch_audio with real dataset"""

def test_fetch_audio_with_dataset():
    """Test with actual HuggingFace dataset"""
    from datasets import load_dataset
    from qwen_vl_utils.vision_process import fetch_audio
    
    print("\n🔍 Testing fetch_audio with real dataset...")
    
    # Load one sample from your dataset
    dataset = load_dataset(
        "speechbrain/LargeScaleASR", 
        "small", 
        split="train", 
        streaming=True
    )
    sample = next(iter(dataset.take(1)))
    
    # Test fetch_audio
    audio_array, sample_rate = fetch_audio(sample["wav"])
    
    # Verify it worked
    print(f"✓ Audio shape: {audio_array.shape}")
    print(f"✓ Sample rate: {sample_rate}Hz")
    print(f"✓ Duration: {len(audio_array)/sample_rate:.2f} seconds")
    print(f"✓ Data type: {audio_array.dtype}")
    
    assert len(audio_array) > 0, "Audio should not be empty"
    assert sample_rate > 0, "Sample rate should be positive"
    print("\n✅ Test passed!")

if __name__ == "__main__":
    test_fetch_audio_with_dataset()