from lib import *
from HuBert_code import HubertEmbeddingExtractor
from Data2Vec_code import Data2VecEmbeddingExtractor
from UniSpeech_code import UniSpeechEmbeddingExtractor
from Whisper_DA import WhisperEmbeddingExtractorDA
from Whisper_Best import WhisperEmbeddingExtractor
from Wav2Vec_code import Wav2Vec2ModelForClassification
from WavLm import WavLMEmbeddingExtractor
from Dataloader import AudioDataset
from classifier import Classifier
from train import train_model
from eval import evaluate_model
from predict_save import predict_labels, save_predictions

def main_pipeline(model_name, train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, 
                  embedding_extractor, epochs=25, batch_size=32):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if embedding_extractor == 'whisper-large':
        extractor = WhisperEmbeddingExtractor(model_name, device=device)
    elif embedding_extractor == 'whisper-da':
        extractor = WhisperEmbeddingExtractorDA(model_name, device=device)
    if embedding_extractor == 'data2vec':
        extractor = Data2VecEmbeddingExtractor(model_name, device=device)        
    elif embedding_extractor in 'hubert':
        extractor = HubertEmbeddingExtractor(model_name, device=device)
    elif embedding_extractor in 'unispeech':
        extractor = UniSpeechEmbeddingExtractor(model_name, device=device)
    elif embedding_extractor in 'wav2vec':
        extractor = Wav2Vec2ModelForClassification(model_name=model_name, device=device).to(device)
    elif embedding_extractor in 'wavlm':
        extractor = WavLMEmbeddingExtractor(model_name, device=device)
    else:
        raise ValueError(f"Invalid embedding extractor: {embedding_extractor}")
        
    train_dataset = AudioDataset(train_csv, train_audio_dir, extractor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    sample_embedding, _ = next(iter(train_loader))

    # Get input dimensions properly, supporting both 1D and 2D tensors
    input_dim = sample_embedding.shape[-1]  # This works for both 1D and 2D tensors

    output_dim = len(pd.read_csv(train_csv).columns) - 1
    classifier = Classifier(input_dim=input_dim, output_dim=output_dim).to(device) #change according to need

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier = train_model(classifier, train_loader, criterion, optimizer, device, epochs)

    criterion = nn.BCEWithLogitsLoss()
    print("Evaluating model...")
    train_loss, train_log_loss, train_accuracy = evaluate_model(classifier, train_loader, criterion, device)
    print(f"Training Loss: {train_loss:.4f}, Log Loss: {train_log_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    test_dataset = AudioDataset(test_csv, test_audio_dir, extractor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = predict_labels(test_loader, classifier, device)

    save_predictions(predictions, output_csv)


if __name__ == "__main__":
    MODEL_NAME = "facebook/data2vec-audio-large-960h" 
    TRAIN_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    TRAIN_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    TEST_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    TEST_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    OUTPUT_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predicted_whisper_large_test_labels.csv"
    EMBEDDING_EXTRACTOR = 'data2vec'  # Change to 'hubert-base' or 'hubert-large' as needed
    
    main_pipeline(MODEL_NAME, TRAIN_CSV, TRAIN_AUDIO_DIR, TEST_CSV, TEST_AUDIO_DIR, OUTPUT_CSV, EMBEDDING_EXTRACTOR)
