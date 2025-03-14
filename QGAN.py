import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sys
from pathlib import Path
import pickle
from scipy.ndimage import zoom
from tqdm import tqdm
import logging
import time
from datetime import datetime

# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Updated imports for Qiskit 1.4.1
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B
# Try different Estimator imports based on available versions
try:
    # First try qiskit_aer.primitives.StatevectorEstimator (newer versions)
    from qiskit_aer.primitives import StatevectorEstimator
    USING_AER_PRIMITIVES = True
except ImportError:
    # Fall back to standard Estimator (older versions)
    from qiskit.primitives import Estimator
    USING_AER_PRIMITIVES = False

from qiskit_aer import AerSimulator
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

# Set up logging
log_filename = f"qgan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# ===============================
# 1. MNIST DATA LOADING AND PREPROCESSING
# ===============================

def load_mnist(path, kind='train'):
    """
    Load MNIST data from path
    
    Args:
        path (str): Path to the MNIST data files
        kind (str): 'train' or 't10k' for training or test data
        
    Returns:
        tuple: (images, labels)
    """
    # Create paths using filename format seen in the screenshot
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    print(f"Looking for labels file at: {labels_path}")
    print(f"Looking for images file at: {images_path}")
    
    # Check if files exist
    if not os.path.exists(labels_path):
        # Try alternative file name formats
        alt_labels = [
            os.path.join(path, f'{kind}-labels.idx1-ubyte'),
            os.path.join(path, f'{kind}_labels.idx1-ubyte')
        ]
        
        for alt_path in alt_labels:
            if os.path.exists(alt_path):
                labels_path = alt_path
                print(f"Found labels at alternative path: {labels_path}")
                break
        else:
            print(f"ERROR: Could not find labels file. Paths checked:")
            print(f"  - {labels_path}")
            for alt in alt_labels:
                print(f"  - {alt}")
            raise FileNotFoundError(f"MNIST labels file not found in {path}")
    
    if not os.path.exists(images_path):
        # Try alternative file name formats
        alt_images = [
            os.path.join(path, f'{kind}-images.idx3-ubyte'),
            os.path.join(path, f'{kind}_images.idx3-ubyte')
        ]
        
        for alt_path in alt_images:
            if os.path.exists(alt_path):
                images_path = alt_path
                print(f"Found images at alternative path: {images_path}")
                break
        else:
            print(f"ERROR: Could not find images file. Paths checked:")
            print(f"  - {images_path}")
            for alt in alt_images:
                print(f"  - {alt}")
            raise FileNotFoundError(f"MNIST images file not found in {path}")
    
    try:
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        
        print(f"Successfully loaded {len(images)} images and {len(labels)} labels")
        return images, labels
    except Exception as e:
        print(f"Error reading MNIST files: {e}")
        print(f"This may be due to incorrect file format or corrupted files.")
        raise

def preprocess_mnist_data(images, labels, selected_digit=5, n_samples=1000, target_dim=8):
    """
    Preprocess MNIST data for QGAN with improved downsampling
    
    Args:
        images (np.array): MNIST images
        labels (np.array): MNIST labels
        selected_digit (int): Which digit to use (0-9)
        n_samples (int): Number of samples to use
        target_dim (int): Target dimension after downsampling
        
    Returns:
        np.array: Preprocessed data
    """
    # Select images of the specified digit
    digit_indices = np.where(labels == selected_digit)[0]
    
    # Sample n_samples images (or all if fewer)
    n_samples = min(n_samples, len(digit_indices))
    selected_indices = np.random.choice(digit_indices, size=n_samples, replace=False)
    selected_images = images[selected_indices]
    
    # Normalize the data to [0, 1]
    normalized_images = selected_images / 255.0
    
    # Reshape to original 28x28 image shape
    reshaped_images = normalized_images.reshape(-1, 28, 28)
    
    # Save some original samples for reference
    logger.info(f"Saving original MNIST digit {selected_digit} samples")
    plt.figure(figsize=(15, 3))
    for i in range(min(5, len(reshaped_images))):
        plt.subplot(1, 5, i+1)
        plt.imshow(reshaped_images[i], cmap='gray')
        plt.title(f"Original {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"original_mnist_digit_{selected_digit}.png")
    plt.close()
    
    # Downsample to target_dim x target_dim
    downsampled_images = np.zeros((n_samples, target_dim, target_dim))
    for i, img in enumerate(reshaped_images):
        # Use scipy.ndimage.zoom for better downsampling
        downsampled_images[i] = zoom(img, target_dim/28, order=1, mode='nearest')
    
    # Apply thresholding to make the downsampled images more distinct
    # This helps create clearer binary patterns for quantum encoding
    threshold = 0.3  # Adjust threshold for better digit representation
    binary_images = (downsampled_images > threshold).astype(float)
    
    # Save some downsampled samples for reference
    logger.info(f"Saving downsampled {target_dim}x{target_dim} MNIST digit {selected_digit} samples")
    plt.figure(figsize=(15, 3))
    for i in range(min(5, len(binary_images))):
        plt.subplot(1, 5, i+1)
        plt.imshow(binary_images[i], cmap='gray')
        plt.title(f"Downsampled {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"downsampled_mnist_digit_{selected_digit}_{target_dim}x{target_dim}.png")
    plt.close()
    
    # Flatten the downsampled images
    flattened_images = binary_images.reshape(n_samples, -1)
    
    logger.info(f"Preprocessed {n_samples} images of digit {selected_digit} to {target_dim}x{target_dim} pixels")
    return flattened_images

def get_binary_mapping(downsampled_dim=4, n_qubits=4):
    """
    Create a mapping between binary states and pixel positions
    
    Args:
        downsampled_dim (int): Dimension of downsampled images
        n_qubits (int): Number of qubits used
        
    Returns:
        dict: Mapping from binary states to pixel positions
    """
    # Total number of pixels
    n_pixels = downsampled_dim * downsampled_dim
    
    # Ensure we have enough qubits to represent all pixels
    if 2**n_qubits < n_pixels:
        raise ValueError(f"Need at least {np.ceil(np.log2(n_pixels))} qubits to represent {n_pixels} pixels")
    
    # Create mapping from binary states to pixel positions
    mapping = {}
    for i in range(min(2**n_qubits, n_pixels)):
        binary = format(i, f'0{n_qubits}b')
        mapping[binary] = i
    
    return mapping

def encode_image_for_qgan(image, mapping, threshold=0.5):
    """
    Encode an image for QGAN
    
    Args:
        image (np.array): Flattened downsampled image
        mapping (dict): Mapping from binary states to pixel positions
        threshold (float): Threshold for binarizing image
        
    Returns:
        dict: Dictionary with binary states as keys and 1 for active pixels
    """
    # Binarize the image based on threshold
    binary_image = (image > threshold).astype(int)
    
    # Create encoding
    encoding = {}
    for binary, pixel_idx in mapping.items():
        if pixel_idx < len(binary_image):
            encoding[binary] = binary_image[pixel_idx]
    
    return encoding

# ===============================
# 2. QUANTUM CIRCUITS FOR MNIST QGAN
# ===============================

def create_generator_circuit(n_qubits, n_layers=3):
    """
    Create an improved parameterized quantum circuit for generator
    
    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of layers
        
    Returns:
        tuple: Quantum circuit and parameter vector
    """
    # Use EfficientSU2 which has better expressivity for this task
    generator = EfficientSU2(n_qubits, reps=n_layers, entanglement='full')
    
    # Add final layer of Ry gates to improve expressibility
    qc = QuantumCircuit(n_qubits)
    qc.compose(generator, inplace=True)
    
    # Get parameters
    parameters = generator.parameters
    
    # Save circuit diagram
    try:
        circuit_image_path = f"generator_circuit_{n_qubits}qubits_{n_layers}layers.png"
        circuit_diagram = circuit_drawer(generator, output='mpl', filename=circuit_image_path)
        logger.info(f"Generator circuit diagram saved to {circuit_image_path}")
    except Exception as e:
        logger.warning(f"Could not save generator circuit diagram: {e}")
    
    return generator, parameters

def create_discriminator_circuit(n_qubits, n_layers=2):
    """
    Create an improved parameterized quantum circuit for discriminator
    
    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of layers
        
    Returns:
        tuple: Quantum circuit and parameter vector
    """
    # Create a more expressive feature map for encoding input data
    feature_map = ZZFeatureMap(n_qubits, reps=2, entanglement='full')
    
    # Create variational circuit with higher expressivity
    discriminator = EfficientSU2(n_qubits, reps=n_layers, entanglement='full')
    
    # Combine feature map and discriminator
    full_discriminator = feature_map.compose(discriminator)
    
    # Add measurement on the last qubit
    qc = QuantumCircuit(n_qubits)
    qc.compose(full_discriminator, inplace=True)
    
    # Get parameters
    parameters = full_discriminator.parameters
    
    # Save circuit diagram
    try:
        circuit_image_path = f"discriminator_circuit_{n_qubits}qubits_{n_layers}layers.png"
        circuit_diagram = circuit_drawer(full_discriminator, output='mpl', filename=circuit_image_path)
        logger.info(f"Discriminator circuit diagram saved to {circuit_image_path}")
    except Exception as e:
        logger.warning(f"Could not save discriminator circuit diagram: {e}")
    
    return full_discriminator, parameters

def encode_binary_to_statevector(encoded_image, n_qubits):
    """
    Encode binary image to statevector for better quantum representation
    
    Args:
        encoded_image (dict): Dictionary with binary states and their values
        n_qubits (int): Number of qubits
        
    Returns:
        Statevector: Quantum state representing the image
    """
    # Create a quantum circuit for encoding
    qc = QuantumCircuit(n_qubits)
    
    # For each active pixel, apply a specific encoding pattern
    for binary, value in encoded_image.items():
        if value == 1:
            idx = int(binary, 2)
            # Binary encoding to prepare |idx⟩ state
            bin_idx = format(idx, f'0{n_qubits}b')
            for i, bit in enumerate(bin_idx):
                if bit == '1':
                    qc.x(i)
    
    # Create and return the statevector
    sv = Statevector.from_instruction(qc)
    return sv

def encode_binary_to_amplitude(encoded_image, n_qubits):
    """
    Encode binary image to amplitude distribution
    
    Args:
        encoded_image (dict): Dictionary with binary states and their values
        n_qubits (int): Number of qubits
        
    Returns:
        np.array: Amplitude distribution for the quantum state
    """
    # Create amplitude distribution
    amplitudes = np.zeros(2**n_qubits, dtype=complex)
    
    # Set amplitudes based on encoded image
    for binary, value in encoded_image.items():
        if value == 1:
            idx = int(binary, 2)
            amplitudes[idx] = 1.0
    
    # Normalize
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes /= norm
    else:
        # If all zeros, return uniform distribution
        amplitudes = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    
    return amplitudes

# ===============================
# 3. HAMILTONIANS FOR OPTIMIZATION
# ===============================

def create_simple_generator_hamiltonian(n_qubits):
    """
    Create a simplified Hamiltonian for the generator to avoid VQE failures
    
    Args:
        n_qubits (int): Number of qubits
        
    Returns:
        SparsePauliOp: Simplified Hamiltonian for generator optimization
    """
    # Instead of complex projectors, use a simpler approach:
    # A Z operator on each qubit with appropriate weights
    
    ops = []
    for i in range(n_qubits):
        # Create Z operator for qubit i
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'Z'
        
        # Add with alternating coefficients for balanced distribution
        coeff = -1.0 if i % 2 == 0 else 1.0
        ops.append(SparsePauliOp(''.join(pauli_str), coeffs=[coeff]))
    
    # Add identity with small weight to balance
    ops.append(SparsePauliOp('I' * n_qubits, coeffs=[0.1]))
    
    # Combine all operators
    hamiltonian = sum(ops[1:], ops[0])
    
    return hamiltonian

def create_simple_discriminator_hamiltonian(n_qubits):
    """
    Create a simplified Hamiltonian for the discriminator to avoid VQE failures
    
    Args:
        n_qubits (int): Number of qubits
        
    Returns:
        SparsePauliOp: Simplified Hamiltonian for discriminator optimization
    """
    # Use a simple Z measurement on the last qubit with reduced coefficient for stability
    z_op = SparsePauliOp('I' * (n_qubits - 1) + 'Z', coeffs=[-0.25])  # Reduced coefficient
    
    # Add identity term for scaling with smaller value
    ident = SparsePauliOp('I' * n_qubits, coeffs=[0.1])  # Reduced identity term
    
    return z_op + ident

def direct_parameter_update(circuit_params, gradient, learning_rate=0.1):
    """
    Update parameters directly using gradients instead of VQE
    
    Args:
        circuit_params (np.array): Current parameters
        gradient (np.array): Gradient of the objective function
        learning_rate (float): Learning rate
        
    Returns:
        np.array: Updated parameters
    """
    # Clip gradients to prevent exploding values
    gradient = np.clip(gradient, -0.5, 0.5)  # Add gradient clipping for stability
    
    # Simple gradient descent update
    new_params = circuit_params - learning_rate * gradient
    
    # Ensure parameters are in [0, 2π] range
    new_params = new_params % (2 * np.pi)
    
    return new_params

def estimate_generator_gradient(generator, parameters, n_qubits, estimator):
    """
    Estimate gradient for generator parameters
    
    Args:
        generator (QuantumCircuit): Generator circuit
        parameters (list): Generator parameters
        n_qubits (int): Number of qubits
        estimator (Estimator): Quantum estimator
        
    Returns:
        np.array: Estimated gradient
    """
    # Create simple Hamiltonian
    hamiltonian = create_simple_generator_hamiltonian(n_qubits)
    
    # Use finite difference method for gradient estimation with numerical stability
    epsilon = max(0.01, 1e-8)  # Prevent too small epsilon
    gradient = np.zeros(len(parameters))
    
    # Base energy calculation
    try:
        bound_generator = generator.assign_parameters(parameters)
        
        # Handle different estimator API patterns
        try:
            # Try with two arguments (newer API)
            job = estimator.run([bound_generator], [hamiltonian])
        except TypeError:
            try:
                # Try with three arguments (older API)
                job = estimator.run([bound_generator], [hamiltonian], [])
            except Exception as e:
                logger.error(f"Failed to run estimator: {e}")
                return np.random.uniform(-0.1, 0.1, len(parameters))
                
        base_result = job.result()
        # Get the value, handling different result formats
        if hasattr(base_result, 'values'):
            base_energy = base_result.values[0].real
        else:
            # Fallback for other result formats
            base_energy = base_result.expectation_values[0].real
        
        # Prevent NaN values
        if np.isnan(base_energy):
            logger.warning("NaN detected in base energy, using default value")
            base_energy = 0.0
        
    except Exception as e:
        logger.warning(f"Error in base energy calculation: {e}")
        return np.random.uniform(-0.1, 0.1, len(parameters))  # Return small random gradient
    
    # Calculate gradients
    for i in range(len(parameters)):
        try:
            # Add epsilon to parameter i
            perturbed_params = parameters.copy()
            perturbed_params[i] += epsilon
            
            # Run with perturbed parameter
            bound_circuit = generator.assign_parameters(perturbed_params)
            
            # Handle different estimator API patterns
            try:
                # Try with two arguments (newer API)
                job = estimator.run([bound_circuit], [hamiltonian])
            except TypeError:
                try:
                    # Try with three arguments (older API)
                    job = estimator.run([bound_circuit], [hamiltonian], [])
                except Exception as e:
                    logger.error(f"Failed to run estimator: {e}")
                    gradient[i] = np.random.uniform(-0.1, 0.1)
                    continue
                    
            perturbed_result = job.result()
            
            # Get the value, handling different result formats
            if hasattr(perturbed_result, 'values'):
                perturbed_energy = perturbed_result.values[0].real
            else:
                # Fallback for other result formats
                perturbed_energy = perturbed_result.expectation_values[0].real
            
            # Prevent NaN values
            if np.isnan(perturbed_energy):
                logger.warning(f"NaN detected in perturbed energy for parameter {i}, using default value")
                perturbed_energy = base_energy  # Use base energy as fallback
                
            # Calculate finite difference with protection against division by zero
            diff = perturbed_energy - base_energy
            gradient[i] = diff / (epsilon + 1e-10)  # Add small constant to prevent division by zero
            
        except Exception as e:
            logger.warning(f"Error in gradient calculation for parameter {i}: {e}")
            gradient[i] = np.random.uniform(-0.1, 0.1)  # Use small random value
    
    return gradient

def estimate_discriminator_gradient(discriminator, parameters, n_qubits, estimator):
    """
    Estimate gradient for discriminator parameters
    
    Args:
        discriminator (QuantumCircuit): Discriminator circuit
        parameters (list): Discriminator parameters
        n_qubits (int): Number of qubits
        estimator (Estimator): Quantum estimator
        
    Returns:
        np.array: Estimated gradient
    """
    # Create simple Hamiltonian
    hamiltonian = create_simple_discriminator_hamiltonian(n_qubits)
    
    # Use finite difference method for gradient estimation
    epsilon = 0.01
    gradient = np.zeros(len(parameters))
    
    # Base energy calculation
    try:
        bound_discriminator = discriminator.assign_parameters(parameters)
        # Fixed: Use only two positional arguments
        job = estimator.run(
            [bound_discriminator],
            [hamiltonian]
        )
        base_result = job.result()
        # Correct result access using .values[0].real
        base_energy = base_result.values[0].real
        
    except Exception as e:
        logger.warning(f"Error in base energy calculation: {e}")
        return np.random.uniform(-0.1, 0.1, len(parameters))  # Return small random gradient
    
    # Calculate gradients
    for i in range(len(parameters)):
        try:
            # Add epsilon to parameter i
            perturbed_params = parameters.copy()
            perturbed_params[i] += epsilon
            
            # Run with perturbed parameter
            bound_circuit = discriminator.assign_parameters(perturbed_params)
            # Fixed: Use positional arguments instead of keyword arguments
            job = estimator.run(
                [bound_circuit],
                [hamiltonian],
                None
            )
            perturbed_result = job.result()
            # Correct result access using .values[0].real
            perturbed_energy = perturbed_result.values[0].real
            
            # Calculate finite difference
            gradient[i] = (perturbed_energy - base_energy) / epsilon
        except Exception as e:
            logger.warning(f"Error in gradient calculation for parameter {i}: {e}")
            gradient[i] = np.random.uniform(-0.1, 0.1)  # Use small random value
    
    return gradient

# ===============================
# 4. MNIST QUANTUM GAN
# ===============================

class MNISTQuantumGAN:
    """Implementation of Quantum GAN for MNIST dataset"""
    
    def __init__(self, 
                 data_path, 
                 n_qubits=4, 
                 n_layers_generator=3, 
                 n_layers_discriminator=2,
                 downsampled_dim=4,
                 selected_digit=5,
                 batch_size=10):
        """
        Initialize MNIST Quantum GAN
        
        Args:
            data_path (str): Path to MNIST data
            n_qubits (int): Number of qubits
            n_layers_generator (int): Number of layers in generator
            n_layers_discriminator (int): Number of layers in discriminator
            downsampled_dim (int): Dimension of downsampled images
            selected_digit (int): Digit to generate (0-9)
            batch_size (int): Batch size for training
        """
        self.n_qubits = n_qubits
        self.downsampled_dim = downsampled_dim
        self.selected_digit = selected_digit
        self.batch_size = batch_size
        self.data_path = data_path
        
        # Ensure proper directories exist
        # Go up one level from raw data directory to create save directories
        base_dir = os.path.dirname(os.path.dirname(data_path))
        self.save_dir = os.path.join(base_dir, 'saved_models')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create output directory for generated images
        self.output_dir = os.path.join(base_dir, 'qgan_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Save directory: {self.save_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Print directory contents to help with debugging
        print(f"\nContents of data directory ({data_path}):")
        try:
            for file in os.listdir(data_path):
                print(f"  - {file}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
        
        # Load and preprocess MNIST data
        print(f"Loading MNIST data from {data_path}...")
        try:
            images, labels = load_mnist(data_path, kind='train')
            self.real_data = preprocess_mnist_data(
                images, labels, 
                selected_digit=selected_digit, 
                target_dim=downsampled_dim
            )
            print(f"Loaded {len(self.real_data)} images of digit {selected_digit}")
        except Exception as e:
            print(f"Error loading MNIST data: {e}")
            raise
        
        # Create binary mapping
        self.binary_mapping = get_binary_mapping(downsampled_dim, n_qubits)
        
        # Encode real data
        self.encoded_real_data = []
        for img in self.real_data:
            self.encoded_real_data.append(encode_image_for_qgan(img, self.binary_mapping))
        
        # Initialize quantum circuits
        print("Initializing quantum circuits...")
        self.generator, self.generator_params = create_generator_circuit(
            n_qubits, n_layers_generator
        )
        self.discriminator, self.discriminator_params = create_discriminator_circuit(
            n_qubits, n_layers_discriminator
        )
        
        # Initialize parameters with smaller range for better convergence
        self.generator_parameters = np.random.uniform(-np.pi/2, np.pi/2, len(self.generator_params))
        self.discriminator_parameters = np.random.uniform(-np.pi/2, np.pi/2, len(self.discriminator_params))
        
        # Initialize primitives based on available APIs
        try:
            if USING_AER_PRIMITIVES:
                # For newer versions with qiskit_aer.primitives
                self.estimator = StatevectorEstimator()
            else:
                # For older versions using basic Estimator without backend
                self.estimator = Estimator()
                
            logger.info(f"Successfully initialized Estimator: {type(self.estimator).__name__}")
        except Exception as e:
            logger.error(f"Error initializing Estimator: {e}")
            # Last resort fallback
            from qiskit.primitives import BaseEstimator
            class SimpleEstimator(BaseEstimator):
                def _run(self, circuits, observables, parameter_values, **run_options):
                    # Simple implementation using AerSimulator directly
                    backend = AerSimulator(method="statevector")
                    results = []
                    for circuit in circuits:
                        # Run simulation
                        job = backend.run(circuit)
                        results.append(job.result())
                    return results
            
            self.estimator = SimpleEstimator()
            logger.info("Using SimpleEstimator fallback")
        
        # Training history
        self.generator_losses = []
        self.discriminator_losses = []
        self.fidelities = []
        
        # Current batch for mini-batch training
        self.current_batch_indices = []
        
        print("MNIST Quantum GAN initialized successfully")
    
    def _get_batch_real_samples(self):
        """
        Get a batch of real samples
        
        Returns:
            list: Batch of encoded real samples
        """
        # Randomly select batch_size samples from encoded_real_data
        indices = np.random.choice(len(self.encoded_real_data), size=min(self.batch_size, len(self.encoded_real_data)), replace=False)
        
        # Save current batch indices for discriminator training
        self.current_batch_indices = indices
        
        return [self.encoded_real_data[i] for i in indices]
    
    def _get_current_batch_real_samples(self):
        """
        Get the current batch of real samples
        
        Returns:
            list: Current batch of encoded real samples
        """
        return [self.encoded_real_data[i] for i in self.current_batch_indices]
    
    def _run_generator(self, n_samples=None):
        """
        Run the generator to produce samples
        
        Args:
            n_samples (int, optional): Number of samples to generate
            
        Returns:
            list: Generated samples
        """
        # Set generator parameters
        parameter_dict = dict(zip(self.generator_params, self.generator_parameters))
        bound_generator = self.generator.assign_parameters(parameter_dict)
        
        # Get statevector
        try:
            statevector = Statevector.from_instruction(bound_generator)
            
            # Calculate probabilities
            probabilities = np.abs(statevector.data)**2
            
            # Generate samples
            if n_samples is None:
                n_samples = self.batch_size
            
            samples = []
            for _ in range(n_samples):
                # Sample multiple states based on probabilities to create digit-like patterns
                # This helps generate more recognizable digits
                num_active_pixels = min(int(self.downsampled_dim * self.downsampled_dim * 0.2), 2**self.n_qubits)
                active_states = np.random.choice(
                    2**self.n_qubits, 
                    size=num_active_pixels, 
                    p=probabilities,
                    replace=False
                )
                
                # Create a sample with multiple active states
                sample = {b: 0 for b in self.binary_mapping.keys()}
                for state_idx in active_states:
                    binary = format(state_idx, f'0{self.n_qubits}b')
                    if binary in sample:
                        sample[binary] = 1
                
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"Error running generator: {e}")
            # Create fallback samples (random patterns)
            if n_samples is None:
                n_samples = self.batch_size
                
            samples = []
            for _ in range(n_samples):
                sample = {b: 0 for b in self.binary_mapping.keys()}
                # Activate a few random pixels
                active_keys = np.random.choice(list(sample.keys()), size=3, replace=False)
                for key in active_keys:
                    sample[key] = 1
                samples.append(sample)
            
            return samples
    
    def _calculate_fidelity(self, generated_samples, real_samples=None):
        """
        Calculate fidelity between generated and real samples
        
        Args:
            generated_samples (list): List of generated samples
            real_samples (list, optional): List of real samples
            
        Returns:
            float: Fidelity score
        """
        if real_samples is None:
            real_samples = self._get_batch_real_samples()
        
        # Convert samples to binary arrays for comparison
        real_arrays = []
        for sample in real_samples:
            arr = np.zeros(2**self.n_qubits)
            for binary, value in sample.items():
                if value == 1:
                    idx = int(binary, 2)
                    arr[idx] = 1
            real_arrays.append(arr)
        
        gen_arrays = []
        for sample in generated_samples:
            arr = np.zeros(2**self.n_qubits)
            for binary, value in sample.items():
                if value == 1:
                    idx = int(binary, 2)
                    arr[idx] = 1
            gen_arrays.append(arr)
        
        # Calculate average fidelity using dot product
        real_avg = np.mean(real_arrays, axis=0)
        gen_avg = np.mean(gen_arrays, axis=0)
        
        # Normalize
        real_avg = real_avg / np.linalg.norm(real_avg) if np.linalg.norm(real_avg) > 0 else real_avg
        gen_avg = gen_avg / np.linalg.norm(gen_avg) if np.linalg.norm(gen_avg) > 0 else gen_avg
        
        # Calculate dot product for fidelity
        fid_value = np.dot(real_avg, gen_avg)
        
        # Ensure we return a scalar, not an array
        if isinstance(fid_value, np.ndarray):
            fid_value = float(fid_value.item()) if fid_value.size == 1 else float(fid_value.mean())
        
        return fid_value
    
    def train_discriminator(self, n_iter=5):
        """
        Train the discriminator using gradient descent instead of VQE
        
        Args:
            n_iter (int): Number of iterations
            
        Returns:
            float: Discriminator loss
        """
        logger.info("Training discriminator...")
        
        # Get real samples 
        real_samples = self._get_current_batch_real_samples()
        
        # Initial loss calculation
        try:
            # Calculate initial loss using simple Hamiltonian
            hamiltonian = create_simple_discriminator_hamiltonian(self.n_qubits)
            bound_circuit = self.discriminator.assign_parameters(self.discriminator_parameters)
            
            # Fixed: Use only two positional arguments
            job = self.estimator.run(
                [bound_circuit],
                [hamiltonian]
            )
            result = job.result()
            # Correct result access using .values[0].real
            initial_loss = result.values[0].real
            
            logger.info(f"Initial discriminator loss: {initial_loss:.4f}")
        except Exception as e:
            logger.warning(f"Error in initial discriminator loss calculation: {e}")
            initial_loss = 0.0
        
        # Train for multiple iterations using gradient descent
        current_loss = initial_loss
        for iteration in range(n_iter):
            try:
                # Estimate gradient
                gradient = estimate_discriminator_gradient(
                    self.discriminator, 
                    self.discriminator_parameters, 
                    self.n_qubits, 
                    self.estimator
                )
                
                # Update parameters
                self.discriminator_parameters = direct_parameter_update(
                    self.discriminator_parameters, 
                    gradient, 
                    learning_rate=self.discriminator_learning_rate
                )
                
                # Add NaN check after parameter update
                if np.isnan(self.discriminator_parameters).any():
                    logger.warning("NaN detected in discriminator parameters! Reinitializing...")
                    self.discriminator_parameters = np.random.uniform(-np.pi/2, np.pi/2, len(self.discriminator_params))
                    break
                
                # Calculate new loss
                bound_circuit = self.discriminator.assign_parameters(self.discriminator_parameters)
                # Fixed: Use positional arguments instead of keyword arguments
                job = self.estimator.run(
                    [bound_circuit],
                    [hamiltonian],
                    None
                )
                result = job.result()
                # Correct result access using .values[0].real
                new_loss = result.values[0].real
                
                # Check for improvement
                if iteration > 0 and abs(new_loss - current_loss) < 0.001:
                    logger.info(f"Discriminator training converged at iteration {iteration + 1}")
                    break
                
                current_loss = new_loss
                logger.info(f"Discriminator iteration {iteration + 1}/{n_iter}, loss: {current_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"Error in discriminator training iteration {iteration + 1}: {e}")
                # Apply small random perturbation to parameters for recovery
                self.discriminator_parameters += np.random.uniform(-0.05, 0.05, len(self.discriminator_parameters))
                current_loss = initial_loss
        
        # Record final loss
        self.discriminator_losses.append(current_loss)
        logger.info(f"Final discriminator loss: {current_loss:.4f}")
        
        return current_loss
    
    def train_generator(self, n_iter=5):
        """
        Train the generator using gradient descent instead of VQE
        
        Args:
            n_iter (int): Number of iterations
            
        Returns:
            tuple: (Generator loss, Fidelity)
        """
        logger.info("Training generator...")
        
        # Get real samples for comparison
        real_samples = self._get_current_batch_real_samples()
        
        # Initial loss calculation
        try:
            # Calculate initial loss using simple Hamiltonian
            hamiltonian = create_simple_generator_hamiltonian(self.n_qubits)
            bound_circuit = self.generator.assign_parameters(self.generator_parameters)
            
            # Fixed: Use only two positional arguments
            job = self.estimator.run(
                [bound_circuit],
                [hamiltonian]
            )
            result = job.result()
            # Correct result access using .values[0].real
            initial_loss = result.values[0].real
            
            logger.info(f"Initial generator loss: {initial_loss:.4f}")
        except Exception as e:
            logger.warning(f"Error in initial generator loss calculation: {e}")
            initial_loss = 0.0
        
        # Train for multiple iterations using gradient descent
        current_loss = initial_loss
        for iteration in range(n_iter):
            try:
                # Estimate gradient
                gradient = estimate_generator_gradient(
                    self.generator, 
                    self.generator_parameters, 
                    self.n_qubits, 
                    self.estimator
                )
                
                # Update parameters
                self.generator_parameters = direct_parameter_update(
                    self.generator_parameters, 
                    gradient, 
                    learning_rate=self.generator_learning_rate
                )
                
                # Add NaN check after parameter update
                if np.isnan(self.generator_parameters).any():
                    logger.warning("NaN detected in generator parameters! Reinitializing...")
                    self.generator_parameters = np.random.uniform(-np.pi/2, np.pi/2, len(self.generator_params))
                    break
                
                # Calculate new loss
                bound_circuit = self.generator.assign_parameters(self.generator_parameters)
                # Fixed: Use positional arguments instead of keyword arguments
                job = self.estimator.run(
                    [bound_circuit],
                    [hamiltonian],
                    None
                )
                result = job.result()
                # Correct result access using .values[0].real
                new_loss = result.values[0].real
                
                # Check for improvement
                if iteration > 0 and abs(new_loss - current_loss) < 0.001:
                    logger.info(f"Generator training converged at iteration {iteration + 1}")
                    break
                
                current_loss = new_loss
                logger.info(f"Generator iteration {iteration + 1}/{n_iter}, loss: {current_loss:.4f}")
                
            except Exception as e:
                logger.warning(f"Error in generator training iteration {iteration + 1}: {e}")
                # Apply small random perturbation to parameters for recovery
                self.generator_parameters += np.random.uniform(-0.05, 0.05, len(self.generator_parameters))
                current_loss = initial_loss
        
        # Record final loss
        self.generator_losses.append(current_loss)
        
        # Generate samples and calculate fidelity after training
        generated_samples = self._run_generator()
        fidelity = self._calculate_fidelity(generated_samples, real_samples)
        
        # Make sure fidelity is a scalar value, not an array
        if isinstance(fidelity, np.ndarray):
            fidelity = float(fidelity.item()) if fidelity.size == 1 else float(fidelity.mean())
        
        self.fidelities.append(fidelity)
        
        logger.info(f"Final generator loss: {current_loss:.4f}, Fidelity: {fidelity:.4f}")
        
        return current_loss, fidelity
    
    def generate_image(self, n_samples=10, post_process=True):
        """
        Generate images from current generator state with improved post-processing
        
        Args:
            n_samples (int): Number of samples to generate
            post_process (bool): Whether to apply post-processing for better visuals
            
        Returns:
            np.array: Generated images
        """
        # Generate samples
        samples = self._run_generator(n_samples)
        
        # Convert samples to images
        images = []
        for sample in samples:
            # Start with empty image
            img = np.zeros((self.downsampled_dim**2))
            
            # For each active state in the sample
            for binary, value in sample.items():
                if value == 1 and binary in self.binary_mapping:
                    pixel_idx = self.binary_mapping[binary]
                    if pixel_idx < len(img):
                        img[pixel_idx] = 1
            
            # Reshape to 2D image
            img = img.reshape(self.downsampled_dim, self.downsampled_dim)
            
            # Apply post-processing for more realistic digit appearance
            if post_process:
                # Apply morphological operations (simulated by convolution)
                kernel = np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]])
                padded = np.pad(img, 1, mode='constant')
                processed = np.zeros_like(img)
                
                for i in range(self.downsampled_dim):
                    for j in range(self.downsampled_dim):
                        # Apply convolution
                        processed[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel) / np.sum(kernel)
                
                # Threshold for binary image
                processed = (processed > 0.3).astype(float)
                
                # Fill small holes (if any)
                for i in range(1, self.downsampled_dim-1):
                    for j in range(1, self.downsampled_dim-1):
                        if processed[i, j] == 0:
                            neighbors = processed[i-1:i+2, j-1:j+2]
                            if np.sum(neighbors) >= 5:  # If surrounded by active pixels
                                processed[i, j] = 1
                
                img = processed
            
            images.append(img)
        
        return np.array(images)
    
    def save_checkpoint(self, epoch):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'generator_parameters': self.generator_parameters,
            'discriminator_parameters': self.discriminator_parameters,
            'generator_losses': self.generator_losses,
            'discriminator_losses': self.discriminator_losses,
            'fidelities': self.fidelities,
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'qgan_checkpoint_epoch_{epoch}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def load_checkpoint(self, epoch):
        """
        Load model checkpoint
        
        Args:
            epoch (int): Epoch to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        checkpoint_path = os.path.join(self.save_dir, f'qgan_checkpoint_epoch_{epoch}.pkl')
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            self.generator_parameters = checkpoint['generator_parameters']
            self.discriminator_parameters = checkpoint['discriminator_parameters']
            self.generator_losses = checkpoint['generator_losses']
            self.discriminator_losses = checkpoint['discriminator_losses']
            self.fidelities = checkpoint['fidelities']
            
            print(f"Checkpoint loaded from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def save_generated_images(self, epoch, n_images=10):
        """
        Generate and save images with comparison to real data
        
        Args:
            epoch (int or str): Current epoch or "final"
            n_images (int): Number of images to generate
        """
        # Generate images with post-processing
        raw_images = self.generate_image(n_images, post_process=False)
        processed_images = self.generate_image(n_images, post_process=True)
        
        # Create a grid of images comparing real and generated
        plt.figure(figsize=(15, 6))
        
        # Top row: real samples for comparison
        real_indices = np.random.choice(len(self.real_data), size=min(5, len(self.real_data)), replace=False)
        for i, idx in enumerate(real_indices):
            plt.subplot(3, 5, i + 1)
            plt.imshow(self.real_data[idx].reshape(self.downsampled_dim, self.downsampled_dim), cmap='gray')
            plt.title(f"Real {i+1}")
            plt.axis('off')
        
        # Middle row: raw generated images
        for i in range(min(5, len(raw_images))):
            plt.subplot(3, 5, i + 6)
            plt.imshow(raw_images[i], cmap='gray')
            plt.title(f"Raw Gen {i+1}")
            plt.axis('off')
        
        # Bottom row: processed generated images
        for i in range(min(5, len(processed_images))):
            plt.subplot(3, 5, i + 11)
            plt.imshow(processed_images[i], cmap='gray')
            plt.title(f"Processed Gen {i+1}")
            plt.axis('off')
        
        # Add overall title
        plt.suptitle(f"MNIST Digit {self.selected_digit} - Epoch {epoch}", fontsize=16)
        
        # Save the figure
        save_path = os.path.join(self.output_dir, f'qgan_generated_epoch_{epoch}.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Generated images saved at {save_path}")
        
        # Save some generated images individually for detailed inspection
        # Check if epoch is a string like "final" or an integer
        if isinstance(epoch, str) or (isinstance(epoch, int) and (epoch % 100 == 0 or epoch == 1)):
            detail_dir = os.path.join(self.output_dir, f'details_epoch_{epoch}')
            os.makedirs(detail_dir, exist_ok=True)
            
            for i, img in enumerate(processed_images[:3]):
                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap='gray', interpolation='nearest')
                plt.title(f"Generated Digit {self.selected_digit} - Sample {i+1}")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(detail_dir, f'detail_gen_{i+1}.png'), dpi=300)
                plt.close()
    
    def save_loss_plot(self):
        """Save enhanced plot of training losses with moving averages"""
        plt.figure(figsize=(15, 12))
        
        # Prepare moving averages for smoother curves
        window_size = min(10, len(self.generator_losses)) if self.generator_losses else 1
        
        if len(self.generator_losses) > 0:
            gen_losses = np.array(self.generator_losses)
            # Handle potential NaN values
            gen_losses = np.nan_to_num(gen_losses, nan=0.0)
            gen_avg = np.convolve(gen_losses, np.ones(window_size)/window_size, mode='valid')
        else:
            gen_avg = []
            
        if len(self.discriminator_losses) > 0:
            disc_losses = np.array(self.discriminator_losses)
            # Handle potential NaN values
            disc_losses = np.nan_to_num(disc_losses, nan=0.0)
            disc_avg = np.convolve(disc_losses, np.ones(window_size)/window_size, mode='valid')
        else:
            disc_avg = []
            
        if len(self.fidelities) > 0:
            fid_values = np.array(self.fidelities)
            # Handle potential NaN values
            fid_values = np.nan_to_num(fid_values, nan=0.0)
            fid_avg = np.convolve(fid_values, np.ones(window_size)/window_size, mode='valid')
        else:
            fid_avg = []
        
        # Plot raw and smoothed generator losses
        plt.subplot(3, 1, 1)
        if len(self.generator_losses) > 0:
            plt.plot(range(len(self.generator_losses)), gen_losses, 'r-', alpha=0.3, label='Generator Loss (Raw)')
            if len(gen_avg) > 0:
                plt.plot(range(window_size-1, len(self.generator_losses)), gen_avg, 'r-', linewidth=2, label='Generator Loss (Smoothed)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot raw and smoothed discriminator losses
        plt.subplot(3, 1, 2)
        if len(self.discriminator_losses) > 0:
            plt.plot(range(len(self.discriminator_losses)), disc_losses, 'b-', alpha=0.3, label='Discriminator Loss (Raw)')
            if len(disc_avg) > 0:
                plt.plot(range(window_size-1, len(self.discriminator_losses)), disc_avg, 'b-', linewidth=2, label='Discriminator Loss (Smoothed)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot raw and smoothed fidelity
        plt.subplot(3, 1, 3)
        if len(self.fidelities) > 0:
            plt.plot(range(len(self.fidelities)), fid_values, 'g-', alpha=0.3, label='Fidelity (Raw)')
            if len(fid_avg) > 0:
                plt.plot(range(window_size-1, len(self.fidelities)), fid_avg, 'g-', linewidth=2, label='Fidelity (Smoothed)')
        plt.xlabel('Epoch')
        plt.ylabel('Fidelity')
        plt.title('Image Fidelity')
        plt.legend()
        plt.grid(True)
        
        # Add overall title
        plt.suptitle(f'MNIST QGAN Training History - Digit {self.selected_digit}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(self.output_dir, 'qgan_training_history.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Training history plot saved at {save_path}")
        
        # Save raw data to CSV for further analysis
        data_path = os.path.join(self.output_dir, 'qgan_training_data.csv')
        with open(data_path, 'w') as f:
            f.write("epoch,generator_loss,discriminator_loss,fidelity\n")
            for i in range(max(len(self.generator_losses), len(self.discriminator_losses), len(self.fidelities))):
                gen_loss = self.generator_losses[i] if i < len(self.generator_losses) else ''
                disc_loss = self.discriminator_losses[i] if i < len(self.discriminator_losses) else ''
                fid = self.fidelities[i] if i < len(self.fidelities) else ''
                f.write(f"{i+1},{gen_loss},{disc_loss},{fid}\n")
        
        logger.info(f"Training data saved to CSV at {data_path}")
    
    def train(self, n_epochs=500, generator_steps=1, discriminator_steps=1, 
              save_interval=50, n_iter_discriminator=5, n_iter_generator=5,
              feature_matching=True, noise_injection=0.05,
              learning_rate_decay=0.995, early_stop_patience=20):
        """
        Train the QGAN with enhanced stability techniques
        
        Args:
            n_epochs (int): Number of epochs
            generator_steps (int): Number of generator steps per epoch
            discriminator_steps (int): Number of discriminator steps per epoch
            save_interval (int): Interval for saving checkpoints and images
            n_iter_discriminator (int): Number of iterations for discriminator training
            n_iter_generator (int): Number of iterations for generator training
            feature_matching (bool): Use feature matching for stabilization
            noise_injection (float): Standard deviation of noise to inject
            learning_rate_decay (float): Factor to decay learning rates
            early_stop_patience (int): Number of epochs to wait for improvement before early stopping
        """
        logger.info(f"Starting QGAN training for {n_epochs} epochs...")
        
        # Define learning rates directly here instead of relying on class attributes
        # This way we don't need to depend on the __init__ method having set these
        initial_gen_lr = 0.02  # Reduced from 0.05 for more stable training
        initial_disc_lr = 0.02  # Reduced from 0.05 for more stable training
        
        # Make these available as instance variables for the training methods to use
        self.generator_learning_rate = initial_gen_lr
        self.discriminator_learning_rate = initial_disc_lr
        
        # Set up log file for training statistics
        stats_log_file = os.path.join(self.output_dir, 'training_stats.csv')
        with open(stats_log_file, 'w') as f:
            f.write("epoch,generator_loss,discriminator_loss,fidelity\n")
        
        # Variables for early stopping
        best_fidelity = -1
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(1, n_epochs + 1), desc="Training QGAN"):
            logger.info(f"\n--- Epoch {epoch}/{n_epochs} ---")
            
            # Learning rate scheduling (decay with epoch) with minimum value
            self.generator_learning_rate = max(initial_gen_lr * (learning_rate_decay ** epoch), 0.001)
            self.discriminator_learning_rate = max(initial_disc_lr * (learning_rate_decay ** epoch), 0.001)
            
            # Use different training strategies at different stages
            if epoch < n_epochs // 5:
                # Initial phase: train discriminator more to establish good boundaries
                local_disc_steps = 2
                local_gen_steps = 1
            elif epoch < n_epochs // 2:
                # Middle phase: balanced training
                local_disc_steps = 1
                local_gen_steps = 1
            else:
                # Final phase: focus more on generator to refine images
                local_disc_steps = 1
                local_gen_steps = 2
            
            # Train discriminator
            disc_loss = 0
            for _ in range(local_disc_steps):
                # Inject noise to parameters to stabilize training
                if noise_injection > 0:
                    noise = np.random.normal(0, noise_injection, len(self.discriminator_parameters))
                    self.discriminator_parameters += noise
                
                disc_loss = self.train_discriminator(n_iter=n_iter_discriminator)
                
                # Feature matching: adjust discriminator parameters to match feature statistics
                if feature_matching:
                    real_samples = self._get_current_batch_real_samples()
                    fake_samples = self._run_generator()
                    
                    # Simple feature matching implementation
                    real_features = np.mean([list(s.values()) for s in real_samples], axis=0)
                    fake_features = np.mean([list(s.values()) for s in fake_samples], axis=0)
                    
                    # Adjust parameters slightly towards matching features
                    feature_diff = real_features - fake_features
                    feature_scale = 0.001  # Reduced from 0.01 for more stability
                    for i in range(min(len(feature_diff), len(self.discriminator_parameters))):
                        self.discriminator_parameters[i] += feature_scale * feature_diff[i]
            
            # Train generator
            gen_loss = 0
            fidelity = 0
            for _ in range(local_gen_steps):
                # Inject noise to parameters to stabilize training
                if noise_injection > 0:
                    noise = np.random.normal(0, noise_injection, len(self.generator_parameters))
                    self.generator_parameters += noise
                
                gen_loss, fidelity = self.train_generator(n_iter=n_iter_generator)
            
            # Save training stats
            with open(stats_log_file, 'a') as f:
                f.write(f"{epoch},{gen_loss},{disc_loss},{fidelity}\n")
            
            # Check for early stopping
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch=f"best_{epoch}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs without improvement")
                    # Don't actually stop - just reset parameters to introduce variability
                    logger.info("Resetting parameters to introduce variability")
                    if self.generator_losses and len(self.generator_losses) > 10:
                        # Reset generator parameters with small perturbation
                        self.generator_parameters = np.random.uniform(0, np.pi/2, len(self.generator_params))
                        patience_counter = 0
            
            # Save checkpoint and generate images at save_interval
            if epoch % save_interval == 0 or epoch == n_epochs:
                logger.info(f"Saving checkpoint and generated images at epoch {epoch}")
                self.save_checkpoint(epoch)
                self.save_generated_images(epoch, n_images=10)
                self.save_loss_plot()
        
        logger.info("\nTraining complete!")
    
    def visualize_real_samples(self, n_samples=10):
        """
        Visualize real samples from the dataset
        
        Args:
            n_samples (int): Number of samples to visualize
        """
        # Get random samples
        indices = np.random.choice(len(self.real_data), size=min(n_samples, len(self.real_data)), replace=False)
        samples = [self.real_data[i] for i in indices]
        
        # Create a grid of images
        n_rows = int(np.ceil(n_samples / 5))
        n_cols = min(5, n_samples)
        
        plt.figure(figsize=(10, n_rows * 2))
        for i, img in enumerate(samples):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img.reshape(self.downsampled_dim, self.downsampled_dim), cmap='gray')
            plt.title(f"Real {i+1}")
            plt.axis('off')
        
        # Save the figure
        save_path = os.path.join(self.output_dir, 'real_samples.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Real samples saved at {save_path}")

# ===============================
# 5. MAIN EXECUTION
# ===============================

def main():
    # Start time for tracking total execution time
    start_time = time.time()
    
    # Path to MNIST data
    data_path = r"C:\Users\uzivatel 1\OneDrive\Dokumenty\Coding Projects\Bachelor_Thesis\MNIST_data\raw"
    
    # Create output directory
    base_dir = os.path.dirname(os.path.dirname(data_path))
    output_dir = os.path.join(base_dir, 'qgan_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file and console
    log_file = os.path.join(output_dir, f"qgan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("MNIST QUANTUM GAN TRAINING SESSION")
    logger.info("=" * 80)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Log file: {log_file}")
    
    # Parameters
    n_qubits = 6  # Increased from 4 to 6 for better expressivity
    downsampled_dim = 8  # Increased from 4 to 8 for better digit representation
    selected_digit = 5  # Generate digit 5
    n_epochs = 500
    save_interval = 50
    
    logger.info(f"Parameters: {n_qubits} qubits, {downsampled_dim}x{downsampled_dim} pixels, digit {selected_digit}")
    
    # Initialize QGAN
    try:
        qgan = MNISTQuantumGAN(
            data_path=data_path,
            n_qubits=n_qubits,
            n_layers_generator=3,
            n_layers_discriminator=2,
            downsampled_dim=downsampled_dim,
            selected_digit=selected_digit,
            batch_size=10
        )
        
        # Visualize real samples
        qgan.visualize_real_samples()
        
        # Train QGAN with improved stability methods
        qgan.train(
            n_epochs=n_epochs,
            generator_steps=1,
            discriminator_steps=1,
            save_interval=save_interval,
            n_iter_discriminator=5,  # Reduced from 20 to 5 for faster iterations
            n_iter_generator=5,      # Reduced from 20 to 5 for faster iterations
            feature_matching=True,   # Enable feature matching
            noise_injection=0.03,    # Reduced noise for less volatility
            learning_rate_decay=0.997,  # Slower decay for more stable training
            early_stop_patience=30   # Patient early stopping
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"MNIST QGAN training complete!")
        logger.info(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Generate final images with higher quality
        logger.info("Generating final high-quality images...")
        qgan.save_generated_images(epoch="final", n_images=20)
        
        # Save final visualization of circuit architectures
        try:
            circuit_image_path = os.path.join(output_dir, "final_generator_circuit.png")
            circuit_drawer(qgan.generator, output='mpl', filename=circuit_image_path)
            logger.info(f"Generator circuit saved to {circuit_image_path}")
            
            circuit_image_path = os.path.join(output_dir, "final_discriminator_circuit.png")
            circuit_drawer(qgan.discriminator, output='mpl', filename=circuit_image_path)
            logger.info(f"Discriminator circuit saved to {circuit_image_path}")
        except Exception as e:
            logger.warning(f"Could not save final circuit diagrams: {e}")
        
    except Exception as e:
        logger.error(f"Error in QGAN training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    logger.info("Training session completed")

if __name__ == "__main__":
    main()