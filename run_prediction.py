"""
Real-World Audio Deepfake Detection Script
Test individual audio files or batches with the trained ensemble model
"""

import os
import yaml
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import librosa
import torchaudio.transforms as T
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedCNN(nn.Module):
    """CNN architecture matching the trained model"""
    
    def __init__(self, input_shape, num_classes=2, dropout=0.5):
        super(AdvancedCNN, self).__init__()
        
        time_dim, freq_dim, channels = input_shape
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


class AudioDeepfakeDetector:
    """Complete inference pipeline for audio deepfake detection"""
    
    def __init__(self, params_file='params.yaml'):
        """Initialize detector with trained models"""
        
        logger.info("Initializing Audio Deepfake Detector...")
        
        # Load parameters
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        
        self.sample_rate = params['preprocess']['sample_rate']
        self.max_length = params['preprocess']['max_audio_length']
        self.n_mfcc = params['features']['mfcc']['n_mfcc']
        self.n_fft = params['features']['mfcc']['n_fft']
        self.hop_length = params['features']['mfcc']['hop_length']
        self.n_mels = params['features']['mfcc']['n_mels']
        
        # Ensemble weights
        self.rf_weight = 0.3
        self.cnn_weight = 0.7
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize feature extractors
        self.init_feature_extractors()
        
        # Load models
        self.load_models()
        
        logger.info("Detector initialized successfully!")
    
    def init_feature_extractors(self):
        """Initialize GPU-accelerated feature extractors"""
        
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels
            }
        ).to(self.device)
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device)
    
    def load_models(self):
        """Load trained Random Forest and CNN models"""
        
        models_path = Path('.')
        
        # Load Random Forest
        logger.info("Loading Random Forest model...")
        rf_path = models_path / 'random_forest_model.pkl'
        
        if not rf_path.exists():
            raise FileNotFoundError(f"Random Forest model not found: {rf_path}")
        
        with open(rf_path, 'rb') as f:
            self.rf_model = pickle.load(f)
        
        logger.info("  Random Forest loaded successfully")
        
        # Load CNN
        logger.info("Loading CNN model...")
        cnn_path = models_path / 'cnn_best_model.pth'
        
        if not cnn_path.exists():
            raise FileNotFoundError(f"CNN model not found: {cnn_path}")
        
        # Initialize CNN with correct input shape
        input_shape = (126, 128, 1)
        self.cnn_model = AdvancedCNN(input_shape, num_classes=2)
        self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        logger.info("  CNN loaded successfully")
    
    def load_audio(self, audio_path):
        """Load audio file (any length)"""
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            duration = len(audio) / sr
            logger.info(f"Loaded audio: {len(audio)} samples ({duration:.2f} seconds), {sr} Hz")
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
        
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None
    
    def split_audio_into_chunks(self, audio, chunk_duration=4.0, overlap=0.5):
        """Split long audio into overlapping chunks"""
        
        chunk_samples = int(chunk_duration * self.sample_rate)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        chunks = []
        
        if len(audio) <= chunk_samples:
            if len(audio) < chunk_samples:
                pad_length = chunk_samples - len(audio)
                audio_padded = np.pad(audio, (0, pad_length), mode='constant')
            else:
                audio_padded = audio
            chunks.append(audio_padded)
            logger.info(f"Audio <= 4s: Created 1 chunk (padded to {chunk_samples} samples)")
        else:
            for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
                chunk = audio[start:start + chunk_samples]
                chunks.append(chunk)
            
            if len(audio) % hop_samples != 0:
                last_chunk = audio[-chunk_samples:]
                chunks.append(last_chunk)
            
            logger.info(f"Created {len(chunks)} chunks with {overlap*100:.0f}% overlap")
        
        return chunks
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features for Random Forest"""
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mfccs = self.mfcc_transform(audio_tensor)
            
            delta_filter = torch.tensor([-1.0, 0.0, 1.0], device=self.device).view(1, 1, 3)
            mfccs_padded = torch.nn.functional.pad(mfccs, (1, 1), mode='replicate')
            
            delta1 = torch.nn.functional.conv1d(
                mfccs_padded.view(-1, 1, mfccs_padded.shape[-1]),
                delta_filter
            ).view(mfccs.shape)
            
            delta1_padded = torch.nn.functional.pad(delta1, (1, 1), mode='replicate')
            delta2 = torch.nn.functional.conv1d(
                delta1_padded.view(-1, 1, delta1_padded.shape[-1]),
                delta_filter
            ).view(mfccs.shape)
            
            features = torch.cat([mfccs, delta1, delta2], dim=1)
            
            mean = torch.mean(features, dim=2)
            std = torch.std(features, dim=2)
            max_val, _ = torch.max(features, dim=2)
            min_val, _ = torch.min(features, dim=2)
            median = torch.median(features, dim=2)[0]
            
            stats = torch.cat([mean, std, max_val, min_val, median], dim=1)
        
        return stats.cpu().numpy().squeeze()
    
    def extract_spectrogram_features(self, audio, target_length=126):
        """Extract mel-spectrogram features for CNN"""
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mel_spec = self.mel_spectrogram(audio_tensor)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            
            spec_min = torch.min(mel_spec_db)
            spec_max = torch.max(mel_spec_db)
            
            if spec_max - spec_min > 0:
                normalized = (mel_spec_db - spec_min) / (spec_max - spec_min)
            else:
                normalized = mel_spec_db - spec_min
            
            features = normalized.permute(0, 2, 1)
            
            current_length = features.shape[1]
            if current_length > target_length:
                features = features[:, :target_length, :]
            elif current_length < target_length:
                pad_length = target_length - current_length
                padding = torch.zeros(1, pad_length, features.shape[2], device=self.device)
                features = torch.cat([features, padding], dim=1)
            
            features = features.unsqueeze(-1)
        
        return features.squeeze(0).cpu().numpy()
    
    def predict_chunk(self, audio_chunk):
        """Predict on single 4-second audio chunk"""
        
        if len(audio_chunk) != self.max_length:
            if len(audio_chunk) > self.max_length:
                audio_chunk = audio_chunk[:self.max_length]
            else:
                pad_length = self.max_length - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, pad_length), mode='constant')
        
        mfcc_features = self.extract_mfcc_features(audio_chunk)
        spec_features = self.extract_spectrogram_features(audio_chunk)
        
        rf_proba = self.rf_model.predict_proba(mfcc_features.reshape(1, -1))[0]
        
        spec_tensor = torch.FloatTensor(spec_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_output = self.cnn_model(spec_tensor)
            cnn_proba = torch.softmax(cnn_output, dim=1).cpu().numpy()[0]
        
        ensemble_proba = self.rf_weight * rf_proba + self.cnn_weight * cnn_proba
        
        return {
            'ensemble_proba': ensemble_proba,
            'rf_proba': rf_proba,
            'cnn_proba': cnn_proba
        }
    
    def aggregate_chunk_predictions(self, chunk_predictions, method='average'):
        """Aggregate predictions from multiple chunks"""
        
        if method == 'average':
            ensemble_proba = np.mean([p['ensemble_proba'] for p in chunk_predictions], axis=0)
            rf_proba = np.mean([p['rf_proba'] for p in chunk_predictions], axis=0)
            cnn_proba = np.mean([p['cnn_proba'] for p in chunk_predictions], axis=0)
            
        elif method == 'majority':
            ensemble_votes = [np.argmax(p['ensemble_proba']) for p in chunk_predictions]
            rf_votes = [np.argmax(p['rf_proba']) for p in chunk_predictions]
            cnn_votes = [np.argmax(p['cnn_proba']) for p in chunk_predictions]
            
            ensemble_proba = np.array([
                1 - ensemble_votes.count(1) / len(ensemble_votes),
                ensemble_votes.count(1) / len(ensemble_votes)
            ])
            rf_proba = np.array([
                1 - rf_votes.count(1) / len(rf_votes),
                rf_votes.count(1) / len(rf_votes)
            ])
            cnn_proba = np.array([
                1 - cnn_votes.count(1) / len(cnn_votes),
                cnn_votes.count(1) / len(cnn_votes)
            ])
            
        elif method == 'max_confidence':
            max_conf_idx = np.argmax([np.max(p['ensemble_proba']) for p in chunk_predictions])
            ensemble_proba = chunk_predictions[max_conf_idx]['ensemble_proba']
            rf_proba = chunk_predictions[max_conf_idx]['rf_proba']
            cnn_proba = chunk_predictions[max_conf_idx]['cnn_proba']
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return {
            'ensemble_proba': ensemble_proba,
            'rf_proba': rf_proba,
            'cnn_proba': cnn_proba
        }
    
    def predict_single(self, audio_path, verbose=True, aggregation_method='average', overlap=0.5):
        """Predict on single audio file"""
        
        if verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {audio_path}")
            logger.info(f"{'='*70}")
        
        audio = self.load_audio(audio_path)
        
        if audio is None:
            return None
        
        chunks = self.split_audio_into_chunks(audio, chunk_duration=4.0, overlap=overlap)
        
        if verbose:
            logger.info(f"Analyzing {len(chunks)} audio chunks...")
        
        chunk_predictions = []
        for i, chunk in enumerate(chunks):
            pred = self.predict_chunk(chunk)
            chunk_predictions.append(pred)
            
            if verbose and len(chunks) > 1:
                chunk_pred = 'BONAFIDE' if np.argmax(pred['ensemble_proba']) == 1 else 'SPOOF'
                chunk_conf = np.max(pred['ensemble_proba']) * 100
                logger.info(f"  Chunk {i+1}/{len(chunks)}: {chunk_pred} ({chunk_conf:.1f}%)")
        
        aggregated = self.aggregate_chunk_predictions(chunk_predictions, method=aggregation_method)
        
        ensemble_pred = np.argmax(aggregated['ensemble_proba'])
        rf_pred = np.argmax(aggregated['rf_proba'])
        cnn_pred = np.argmax(aggregated['cnn_proba'])
        
        results = {
            'file': str(audio_path),
            'duration_seconds': len(audio) / self.sample_rate,
            'num_chunks': len(chunks),
            'aggregation_method': aggregation_method,
            'prediction': 'BONAFIDE' if ensemble_pred == 1 else 'SPOOF',
            'confidence': float(aggregated['ensemble_proba'][ensemble_pred]),
            'probabilities': {
                'bonafide': float(aggregated['ensemble_proba'][1]),
                'spoof': float(aggregated['ensemble_proba'][0])
            },
            'individual_models': {
                'random_forest': {
                    'prediction': 'BONAFIDE' if rf_pred == 1 else 'SPOOF',
                    'confidence': float(aggregated['rf_proba'][rf_pred]),
                    'probabilities': {
                        'bonafide': float(aggregated['rf_proba'][1]),
                        'spoof': float(aggregated['rf_proba'][0])
                    }
                },
                'cnn': {
                    'prediction': 'BONAFIDE' if cnn_pred == 1 else 'SPOOF',
                    'confidence': float(aggregated['cnn_proba'][cnn_pred]),
                    'probabilities': {
                        'bonafide': float(aggregated['cnn_proba'][1]),
                        'spoof': float(aggregated['cnn_proba'][0])
                    }
                }
            },
            'chunk_analysis': {
                'total_chunks': len(chunks),
                'spoof_chunks': sum(1 for p in chunk_predictions if np.argmax(p['ensemble_proba']) == 0),
                'bonafide_chunks': sum(1 for p in chunk_predictions if np.argmax(p['ensemble_proba']) == 1)
            }
        }
        
        if verbose:
            self.print_results(results)
        
        return results
    
    def print_results(self, results):
        """Print detection results"""
        
        print(f"\n{'='*80}")
        print(f"{'DETECTION RESULTS':^80}")
        print(f"{'='*80}")
        
        print(f"\n{'File:':<20} {Path(results['file']).name}")
        print(f"{'Duration:':<20} {results['duration_seconds']:.2f} seconds")
        print(f"{'Chunks:':<20} {results['num_chunks']} ({results['aggregation_method']} aggregation)")
        
        if results['num_chunks'] > 1:
            chunk_analysis = results['chunk_analysis']
            print(f"{'Chunk Analysis:':<20} {chunk_analysis['spoof_chunks']} Spoof / {chunk_analysis['bonafide_chunks']} Bonafide")
        
        print(f"\n{'-'*80}")
        print(f"{'INDIVIDUAL MODEL PREDICTIONS':^80}")
        print(f"{'-'*80}")
        
        # Header
        print(f"\n{'Model':<20} {'Prediction':<15} {'Confidence':<15} {'Bonafide %':<15} {'Spoof %':<15}")
        print(f"{'-'*80}")
        
        # Random Forest
        rf = results['individual_models']['random_forest']
        rf_pred_color = '\033[92m' if rf['prediction'] == 'BONAFIDE' else '\033[91m'
        reset = '\033[0m'
        print(f"{'Random Forest':<20} {rf_pred_color}{rf['prediction']:<15}{reset} {rf['confidence']*100:>6.2f}%{' ':<7} {rf['probabilities']['bonafide']*100:>6.2f}%{' ':<7} {rf['probabilities']['spoof']*100:>6.2f}%")
        
        # CNN
        cnn = results['individual_models']['cnn']
        cnn_pred_color = '\033[92m' if cnn['prediction'] == 'BONAFIDE' else '\033[91m'
        print(f"{'CNN':<20} {cnn_pred_color}{cnn['prediction']:<15}{reset} {cnn['confidence']*100:>6.2f}%{' ':<7} {cnn['probabilities']['bonafide']*100:>6.2f}%{' ':<7} {cnn['probabilities']['spoof']*100:>6.2f}%")
        
        # Ensemble
        pred = results['prediction']
        conf = results['confidence'] * 100
        ens_pred_color = '\033[92m' if pred == 'BONAFIDE' else '\033[91m'
        print(f"{'-'*80}")
        print(f"{'ENSEMBLE (Final)':<20} {ens_pred_color}{pred:<15}{reset} {conf:>6.2f}%{' ':<7} {results['probabilities']['bonafide']*100:>6.2f}%{' ':<7} {results['probabilities']['spoof']*100:>6.2f}%")
        
        print(f"\n{'='*80}")
        
        # Model agreement indicator
        rf_pred = results['individual_models']['random_forest']['prediction']
        cnn_pred = results['individual_models']['cnn']['prediction']
        
        if rf_pred == cnn_pred == pred:
            print(f"{'Model Agreement:':<20} All models agree - {pred}")
        else:
            print(f"{'Model Disagreement:':<20} RF={rf_pred}, CNN={cnn_pred}, Final={pred}")
        
        # Confidence indicator
        if conf > 90:
            print(f"{'Confidence Level:':<20} Very High - Clear {pred.lower()} characteristics")
        elif conf > 75:
            print(f"{'Confidence Level:':<20} High - Likely {pred.lower()}")
        elif conf > 60:
            print(f"{'Confidence Level:':<20} Moderate - Probably {pred.lower()}")
        else:
            print(f"{'Confidence Level:':<20} Low - Uncertain detection")
        
        print(f"{'='*80}\n")
    
    def predict_batch(self, audio_dir, output_file=None, aggregation_method='average', overlap=0.5):
        """Predict on batch of audio files"""
        
        audio_dir = Path(audio_dir)
        
        if not audio_dir.exists():
            logger.error(f"Directory not found: {audio_dir}")
            return None
        
        audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(audio_dir.glob(f'*{ext}')))
        
        if not audio_files:
            logger.error(f"No audio files found in {audio_dir}")
            return None
        
        logger.info(f"\nFound {len(audio_files)} audio files")
        logger.info(f"Processing batch...\n")
        
        results = []
        
        for audio_file in audio_files:
            result = self.predict_single(
                audio_file, 
                verbose=False, 
                aggregation_method=aggregation_method,
                overlap=overlap
            )
            if result:
                results.append(result)
                
                pred = result['prediction']
                conf = result['confidence'] * 100
                duration = result['duration_seconds']
                logger.info(f"{audio_file.name:<50} {duration:>6.1f}s  {pred:<10} {conf:>6.2f}%")
        
        if output_file:
            import json
            output_path = Path(output_file)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\nResults saved to: {output_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info("BATCH SUMMARY")
        logger.info(f"{'='*70}")
        
        total = len(results)
        bonafide_count = sum(1 for r in results if r['prediction'] == 'BONAFIDE')
        spoof_count = total - bonafide_count
        
        logger.info(f"\nTotal files: {total}")
        logger.info(f"Bonafide: {bonafide_count} ({bonafide_count/total*100:.1f}%)")
        logger.info(f"Spoof: {spoof_count} ({spoof_count/total*100:.1f}%)")
        
        logger.info(f"{'='*70}\n")
        
        return results


def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(
        description='Audio Deepfake Detection',
        epilog="""
Examples:
  python run_prediction.py
  python run_prediction.py --file bonafide_1.flac
  python run_prediction.py --dir . --output results.json
        """
    )
    
    parser.add_argument('--file', '-f', type=str, help='Path to single audio file')
    parser.add_argument('--dir', '-d', type=str, help='Path to directory containing audio files')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file for batch results')
    parser.add_argument('--aggregation', '-a', type=str, default='average', choices=['average', 'majority', 'max_confidence'])
    parser.add_argument('--overlap', type=float, default=0.5)
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("Audio Deepfake Detector")
    logger.info("="*70 + "\n")
    
    detector = AudioDeepfakeDetector()
    
    # Interactive mode if no arguments provided
    if not args.file and not args.dir:
        logger.info("\n" + "="*70)
        logger.info("INTERACTIVE MODE")
        logger.info("="*70)
        
        while True:
            print("\nOptions:")
            print("  1. Test single audio file")
            print("  2. Test all files in directory")
            print("  3. Quit")
            
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == '1':
                filename = input("\nEnter audio filename (or path): ").strip()
                if filename:
                    if Path(filename).exists():
                        result = detector.predict_single(filename, aggregation_method=args.aggregation, overlap=args.overlap)
                    else:
                        logger.error(f"File not found: {filename}")
                else:
                    logger.warning("No filename provided")
                    
                # Ask if want to continue
                cont = input("\nTest another file? (y/n): ").strip().lower()
                if cont != 'y':
                    break
                    
            elif choice == '2':
                directory = input("\nEnter directory path (. for current): ").strip()
                if not directory:
                    directory = '.'
                    
                output = input("Save results to JSON? (filename or press Enter to skip): ").strip()
                output = output if output else None
                
                results = detector.predict_batch(directory, output, aggregation_method=args.aggregation, overlap=args.overlap)
                break
                
            elif choice == '3':
                logger.info("Exiting...")
                break
            else:
                logger.warning("Invalid choice. Please enter 1, 2, or 3")
    
    elif args.file:
        result = detector.predict_single(args.file, aggregation_method=args.aggregation, overlap=args.overlap)
    elif args.dir:
        results = detector.predict_batch(args.dir, args.output, aggregation_method=args.aggregation, overlap=args.overlap)
    
    logger.info("\nDetection completed!")


if __name__ == "__main__":
    main()
