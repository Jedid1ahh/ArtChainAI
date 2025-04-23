# ArtChainAI - A platform connecting artists, AI, and blockchain for NFT creation and royalty distribution
# Main implementation file

import os
import json
import uuid
import hashlib
import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple

# AI/ML libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Web3 and blockchain integration
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import Address
import solcx
from hexbytes import HexBytes

# Web framework
from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("artchain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ArtChainAI")

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///artchain.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Blockchain configuration
BLOCKCHAIN_PROVIDER = os.environ.get('BLOCKCHAIN_PROVIDER', 'https://rpc-mainnet.matic.network')
CHAIN_ID = int(os.environ.get('CHAIN_ID', 137))  # Polygon Mainnet
CONTRACT_ADDRESS = os.environ.get('CONTRACT_ADDRESS')
PRIVATE_KEY = os.environ.get('PRIVATE_KEY')

# Initialize Web3 connection
w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_PROVIDER))
if PRIVATE_KEY:
    account: LocalAccount = Account.from_key(PRIVATE_KEY)
    logger.info(f"Connected to blockchain with account: {account.address}")
else:
    logger.warning("No private key provided. Some blockchain operations will be limited.")

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize login manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    wallet_address = db.Column(db.String(42), unique=True)
    is_artist = db.Column(db.Boolean, default=False)
    profile_image = db.Column(db.String(256))
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    artworks = db.relationship('Artwork', backref='artist', lazy='dynamic')
    ai_models = db.relationship('AIModel', backref='owner', lazy='dynamic')
    nfts = db.relationship('NFT', backref='creator', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'wallet_address': self.wallet_address,
            'is_artist': self.is_artist,
            'profile_image': self.profile_image,
            'bio': self.bio,
            'created_at': self.created_at.isoformat(),
        }


class Artwork(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    image_path = db.Column(db.String(256), nullable=False)
    artist_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    tags = db.Column(db.String(256))  # Comma-separated tags
    is_used_for_training = db.Column(db.Boolean, default=False)
    content_hash = db.Column(db.String(64), unique=True)  # SHA-256 hash of the image

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'image_path': self.image_path,
            'artist_id': self.artist_id,
            'artist_username': User.query.get(self.artist_id).username,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags.split(',') if self.tags else [],
            'is_used_for_training': self.is_used_for_training,
            'content_hash': self.content_hash,
        }


class AIModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    model_type = db.Column(db.String(64), nullable=False)  # GAN, Diffusion, etc.
    model_path = db.Column(db.String(256))
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    last_trained = db.Column(db.DateTime)
    training_status = db.Column(db.String(32), default='pending')  # pending, training, completed, failed
    training_params = db.Column(db.Text)  # JSON string of training parameters
    training_artworks = db.relationship('Artwork', secondary='model_training_data')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'owner_id': self.owner_id,
            'owner_username': User.query.get(self.owner_id).username,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'training_status': self.training_status,
            'training_params': json.loads(self.training_params) if self.training_params else {},
            'artwork_count': len(self.training_artworks),
        }


# Association table for model training data
model_training_data = db.Table(
    'model_training_data',
    db.Column('model_id', db.Integer, db.ForeignKey('ai_model.id'), primary_key=True),
    db.Column('artwork_id', db.Integer, db.ForeignKey('artwork.id'), primary_key=True),
    db.Column('added_at', db.DateTime, default=datetime.datetime.utcnow)
)


class NFT(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    image_path = db.Column(db.String(256), nullable=False)
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    token_id = db.Column(db.String(256), unique=True)
    contract_address = db.Column(db.String(42))
    blockchain = db.Column(db.String(32), default='polygon')
    status = db.Column(db.String(32), default='pending')  # pending, minted, sold, transferred
    price = db.Column(db.Float)
    currency = db.Column(db.String(10), default='MATIC')
    metadata_uri = db.Column(db.String(256))
    content_hash = db.Column(db.String(64), unique=True)
    ai_model_id = db.Column(db.Integer, db.ForeignKey('ai_model.id'))
    original_artwork_ids = db.Column(db.Text)  # Comma-separated IDs of artworks used for generation

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'image_path': self.image_path,
            'creator_id': self.creator_id,
            'creator_username': User.query.get(self.creator_id).username,
            'created_at': self.created_at.isoformat(),
            'token_id': self.token_id,
            'contract_address': self.contract_address,
            'blockchain': self.blockchain,
            'status': self.status,
            'price': self.price,
            'currency': self.currency,
            'metadata_uri': self.metadata_uri,
            'content_hash': self.content_hash,
            'ai_model_id': self.ai_model_id,
            'original_artwork_ids': [int(id) for id in self.original_artwork_ids.split(',')] if self.original_artwork_ids else [],
        }


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nft_id = db.Column(db.Integer, db.ForeignKey('nft.id'), nullable=False)
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    buyer_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    transaction_hash = db.Column(db.String(66), unique=True)
    status = db.Column(db.String(32), default='pending')  # pending, completed, failed
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(10), default='MATIC')
    transaction_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    royalties_paid = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'nft_id': self.nft_id,
            'seller_id': self.seller_id,
            'seller_username': User.query.get(self.seller_id).username,
            'buyer_id': self.buyer_id,
            'buyer_username': User.query.get(self.buyer_id).username if self.buyer_id else None,
            'transaction_hash': self.transaction_hash,
            'status': self.status,
            'amount': self.amount,
            'currency': self.currency,
            'transaction_date': self.transaction_date.isoformat(),
            'royalties_paid': self.royalties_paid,
        }


class Collaboration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    status = db.Column(db.String(32), default='draft')  # draft, active, completed
    royalty_distribution = db.Column(db.Text)  # JSON string with user_id: percentage pairs

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'creator_id': self.creator_id,
            'creator_username': User.query.get(self.creator_id).username,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'royalty_distribution': json.loads(self.royalty_distribution) if self.royalty_distribution else {},
        }


# Association table for collaborations and models
collaboration_models = db.Table(
    'collaboration_models',
    db.Column('collaboration_id', db.Integer, db.ForeignKey('collaboration.id'), primary_key=True),
    db.Column('model_id', db.Integer, db.ForeignKey('ai_model.id'), primary_key=True),
    db.Column('added_at', db.DateTime, default=datetime.datetime.utcnow)
)


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


# Utility functions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def hash_file(file_path):
    """Generate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_uploaded_file(file):
    """Save an uploaded file and return the path"""
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    return file_path, hash_file(file_path)


# AI Model implementations

class StyleTransferModel:
    """Neural style transfer model using TensorFlow"""

    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()

    def _build_model(self):
        """Initialize a new model based on VGG19"""
        # Use VGG19 as the base model for style transfer
        self.base_model = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False

        # Select layers for content and style representation
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                            'block4_conv1', 'block5_conv1']

        # Build the model
        self.model = self._build_style_model()
        logger.info("Successfully built style transfer model based on VGG19")

    def _build_style_model(self):
        """Build the style transfer model architecture"""
        # Get content and style outputs
        outputs = [self.base_model.get_layer(name).output for name in 
                  self.style_layers + self.content_layers]

        # Create model
        model = tf.keras.Model(inputs=self.base_model.input, outputs=outputs)
        return model

    def save_model(self, path):
        """Save the model to disk"""
        if self.model:
            self.model.save(path)
            logger.info(f"Style transfer model saved to {path}")
            return True
        return False

    def load_model(self, path):
        """Load model from disk"""
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Style transfer model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def process_image(self, img_path):
        """Preprocess an image for the model"""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [512, 512])
        img = img[tf.newaxis, :]
        return img

    def generate_image(self, content_img_path, style_img_path, output_path, iterations=1000):
        """Generate a new image combining content and style"""
        content_image = self.process_image(content_img_path)
        style_image = self.process_image(style_img_path)

        # Create optimizer
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # Initialize generated image with content image
        generated_image = tf.Variable(content_image)

        # Optimization loop
        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Get outputs
                outputs = self.model(generated_image)

                # Extract style and content features
                style_features = outputs[:len(self.style_layers)]
                content_features = outputs[len(self.style_layers):]

                # Get style and content targets
                style_targets = self.model(style_image)[:len(self.style_layers)]
                content_targets = self.model(content_image)[len(self.style_layers):]

                # Calculate losses
                style_loss = self._style_loss(style_features, style_targets)
                content_loss = self._content_loss(content_features, content_targets)

                # Total loss
                total_loss = style_loss * 1e-2 + content_loss

            # Compute gradients and apply updates
            grads = tape.gradient(total_loss, generated_image)
            opt.apply_gradients([(grads, generated_image)])

            # Clip pixel values
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

            if i % 100 == 0:
                logger.info(f"Iteration {i}: style loss={style_loss:.4f}, content loss={content_loss:.4f}")

        # Save the generated image
        result = generated_image[0].numpy()
        result = (result * 255).astype(np.uint8)
        Image.fromarray(result).save(output_path)
        logger.info(f"Generated image saved to {output_path}")

        return output_path

    def _gram_matrix(self, input_tensor):
        """Calculate Gram matrix for style representation"""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def _style_loss(self, style_outputs, style_targets):
        """Calculate style loss"""
        loss = tf.add_n([tf.reduce_mean((self._gram_matrix(style_output) - 
                                         self._gram_matrix(style_target))**2)
                         for style_output, style_target in 
                         zip(style_outputs, style_targets)])
        return loss

    def _content_loss(self, content_outputs, content_targets):
        """Calculate content loss"""
        loss = tf.add_n([tf.reduce_mean((content_output - content_target)**2)
                         for content_output, content_target in 
                         zip(content_outputs, content_targets)])
        return loss


class GANModel:
    """Generative Adversarial Network for art generation using PyTorch"""

    def __init__(self, model_path=None, latent_dim=100, img_size=512):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = None
        self.discriminator = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()

    def _build_model(self):
        """Initialize a new GAN model"""
        # Define generator
        self.generator = nn.Sequential(
            # Input is latent vector z
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3 * self.img_size * self.img_size),
            nn.Tanh()  # Output pixel values in range [-1, 1]
        ).to(self.device)

        # Define discriminator
        self.discriminator = nn.Sequential(
            # Input is flattened image
            nn.Linear(3 * self.img_size * self.img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability that input is real image
        ).to(self.device)

        logger.info(f"Successfully built GAN model on {self.device}")

    def save_model(self, path):
        """Save the model to disk"""
        if self.generator and self.discriminator:
            torch.save({
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'latent_dim': self.latent_dim,
                'img_size': self.img_size
            }, path)
            logger.info(f"GAN model saved to {path}")
            return True
        return False

    def load_model(self, path):
        """Load model from disk"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Update model parameters
            self.latent_dim = checkpoint.get('latent_dim', self.latent_dim)
            self.img_size = checkpoint.get('img_size', self.img_size)

            # Rebuild model architecture if needed
            if self.generator is None or self.discriminator is None:
                self._build_model()

            # Load state dicts
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])

            logger.info(f"GAN model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def train(self, dataloader, epochs=100, lr=0.0002, beta1=0.5):
        """Train the GAN model"""
        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        # Loss function
        adversarial_loss = nn.BCELoss()

        # Training loop
        for epoch in range(epochs):
            for i, (images, _) in enumerate(dataloader):
                # Configure input
                real_images = images.to(self.device)
                real_images = real_images.view(real_images.size(0), -1)
                batch_size = real_images.size(0)

                # Labels
                real_label = torch.ones(batch_size, 1).to(self.device)
                fake_label = torch.zeros(batch_size, 1).to(self.device)

                # -----------------
                # Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of images
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(z)

                # Train on fake images
                g_loss = adversarial_loss(self.discriminator(fake_images), real_label)

                g_loss.backward()
                optimizer_G.step()

                # -----------------
                # Train Discriminator
                # -----------------
                optimizer_D.zero_grad()

                # Train on real images
                real_loss = adversarial_loss(self.discriminator(real_images), real_label)

                # Train on fake images
                fake_loss = adversarial_loss(self.discriminator(fake_images.detach()), fake_label)

                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                if i % 100 == 0:
                    logger.info(
                        f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                    )

            # Save model checkpoint
            if epoch % 10 == 0:
                self.save_model(f"model_checkpoints/gan_epoch_{epoch}.pt")

    def generate_image(self, output_path, num_images=1, seed=None):
        """Generate new images using the trained generator"""
        if seed is not None:
            torch.manual_seed(seed)

        self.generator.eval()

        # Generate random noise vectors
        z = torch.randn(num_images, self.latent_dim).to(self.device)

        # Generate images
        with torch.no_grad():
            fake_images = self.generator(z)

        # Convert to numpy arrays and reshape
        fake_images = fake_images.cpu().numpy()
        fake_images = fake_images.reshape(num_images, 3, self.img_size, self.img_size)

        # Rescale values to [0, 255]
        fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)

        # Save images
        for i in range(num_images):
            img = fake_images[i].transpose(1, 2, 0)  # CHW -> HWC
            Image.fromarray(img).save(f"{output_path}_{i}.png")

        logger.info(f"Generated {num_images} images saved to {output_path}")

        return [f"{output_path}_{i}.png" for i in range(num_images)]


class DiffusionModel:
    """Diffusion model for art generation using PyTorch"""

    def __init__(self, model_path=None, img_size=256, time_steps=1000):
        self.img_size = img_size
        self.time_steps = time_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.beta_schedule = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
            self._setup_diffusion_parameters()

    def _build_model(self):
        """Initialize a simple UNet model for diffusion"""
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=3, time_embedding_dim=256):
                super().__init__()
                self.time_embedding_dim = time_embedding_dim

                # Time embedding
                self.time_embed = nn.Sequential(
                    nn.Linear(1, time_embedding_dim),
                    nn.SiLU(),
                    nn.Linear(time_embedding_dim, time_embedding_dim),
                )

                # Downsampling
                self.down1 = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                self.down2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
                self.down3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )

                # Middle
                self.mid = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )

                # Upsampling
                self.up1 = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                self.up2 = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                self.up3 = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )

                # Output
                self.output = nn.Conv2d(64, in_channels, kernel_size=1)

                # Time projection layers
                self.time_proj1 = nn.Linear(time_embedding_dim, 64)
                self.time_proj2 = nn.Linear(time_embedding_dim, 128)
                self.time_proj3 = nn.Linear(time_embedding_dim, 256)
                self.time_proj4 = nn.Linear(time_embedding_dim, 512)

            def forward(self, x, t):
                # Embed time
                t = t.float().view(-1, 1)
                t = self.time_embed(t)

                # Down
                x1 = self.down1(x)
                x1 = x1 + self.time_proj1(t).view(-1, 64, 1, 1)

                x2 = self.down2(x1)
                x2 = x2 + self.time_proj2(t).view(-1, 128, 1, 1)

                x3 = self.down3(x2)
                x3 = x3 + self.time_proj3(t).view(-1, 256, 1, 1)

                # Middle
                x_mid = self.mid(x3)
                x_mid = x_mid + self.time_proj4(t).view(-1, 512, 1, 1)

                # Up
                x = self.up1(x_mid)
                x = self.up2(x)
                x = self.up3(x)

                # Output
                return self.output(x)

        self.model = SimpleUNet().to(self.device)
        logger.info(f"Successfully built diffusion model on {self.device}")

    def _setup_diffusion_parameters(self):
        """Set up beta schedule and derived diffusion parameters"""
        # Linear beta schedule
        beta_start = 1e-4
        beta_end = 2e-2
        self.beta_schedule = torch.linspace(beta_start, beta_end, self.time_steps).to(self.device)

        # Derived parameters
        self.alpha_schedule = 1 - self.beta_schedule
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)

        logger.info("Diffusion parameters initialized")

    def save_model(self, path):
        """Save the model to disk"""
        if self.model and self.beta_schedule is not None:
            torch.save({
                'model': self.model.state_dict(),
                'beta_schedule': self.beta_schedule,
                'img_size': self.img_size,
                'time_steps': self.time_steps
            }, path)
            logger.info(f"Diffusion model saved to {path}")
            return True
        return False

    def load_model(self, path):
        """Load model from disk"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Update model parameters
            self.img_size = checkpoint.get('img_size', self.img_size)
            self.time_steps = checkpoint.get('time_steps', self.time_steps)

            # Rebuild model architecture if needed
            if self.model is None:
                self._build_model()

            # Load state dict
            self.model.load_state_dict(checkpoint['model'])

            # Load diffusion parameters
            self.beta_schedule = checkpoint.get('beta_schedule', None)
            if self.beta_schedule is None:
                self._setup_diffusion_parameters()
            else:
                # Recompute derived parameters
                self.alpha_schedule = 1 - self.beta_schedule
                self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)

            logger.info(f"Diffusion model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def _add_noise(self, x, t):
        """Add noise to the input"""
        alpha_bar = self.alpha_bar_schedule[t]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)

        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

        return noisy_x, noise

    def train(self, dataloader, epochs=100, lr=1e-4):
        """Train the diffusion model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0

            for i, (images, _) in enumerate(dataloader):
                # Move images to device
                images = images.to(self.device)

                # Sample random timesteps
                batch_size = images.shape[0]
                t = torch.randint(0, self.time_steps, (batch_size,), device=self.device)

                # Add noise to images
                noisy_images, target_noise = self._add_noise(images, t)

                # Predict noise
                predicted_noise = self.model(noisy_images, t)

                # Calculate loss
                loss = mse_loss(predicted_noise, target_noise)

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 100 == 0:
                    logger.info(
                        f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[Loss: {loss.item():.4f}]"
                    )

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")

            # Save model checkpoint
            if epoch % 10 == 0:
                self.save_model(f"model_checkpoints/diffusion_epoch_{epoch}.pt")

    def sample(self, batch_size=1, seed=None):
        """Sample new images from the trained diffusion model"""
        if seed is not None:
            torch.manual_seed(seed)

        self.model.eval()

        # Start with random noise
        x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(self.device)

        # Iteratively denoise
        for t in reversed(range(self.time_steps)):
            t_tensor = torch.tensor([t] * batch_size, device=self.device)

            with torch.no_grad():
                # Predict noise
                predicted_noise = self.model(x, t_tensor)

                # Get parameters for this timestep
                alpha = self.alpha_schedule[t]
                alpha_bar = self.alpha_bar_schedule[t]
                beta = self.beta_schedule[t]

                # No noise for step 0
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

                # Update x
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
                ) + torch.sqrt(beta) * noise

        # Rescale to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2

        return x

    def generate_image(self, output_path, num_images=1, seed=None):
        """Generate new images and save them to disk"""
        # Sample images
        samples = self.sample(batch_size=num_images, seed=seed)

        # Convert to numpy arrays
        samples = samples.cpu().numpy()

        # Save images
        output_paths = []
        for i in range(num_images):
            # Convert from CHW to HWC and scale to [0, 255]
            img = samples[i].transpose(1, 2, 0) * 255
            img = img.astype(np.uint8)

            # Save image
            img_path = f"{output_path}_{i}.png"
            Image.fromarray(img).save(img_path)
            output_paths.append(img_path)

        logger.info(f"Generated {num_images} images saved to {output_path}")

        return output_paths


# Blockchain integration

class SmartContractManager:
    """Manage interactions with blockchain smart contracts"""

    def __init__(self, provider_url=BLOCKCHAIN_PROVIDER, private_key=PRIVATE_KEY):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))

        # Initialize account from private key if provided
        self.account = None
        if private_key:
            self.account = Account.from_key(private_key)
            logger.info(f"Initialized blockchain account: {self.account.address}")
        else:
            logger.warning("No private key provided. Limited functionality available.")

        # Contracts
        self.nft_contract = None
        self.marketplace_contract = None
        self.royalty_contract = None

        # Contract ABIs and bytecode
        self._load_contract_definitions()

    def _load_contract_definitions(self):
        """Load contract ABIs and bytecode"""
        try:
            # NFT Contract
            with open('contracts/ArtChainNFT.json', 'r') as f:
                nft_contract_json = json.load(f)
                self.nft_contract_abi = nft_contract_json['abi']
                self.nft_contract_bytecode = nft_contract_json['bytecode']

            # Marketplace Contract
            with open('contracts/ArtChainMarketplace.json', 'r') as f:
                marketplace_contract_json = json.load(f)
                self.marketplace_contract_abi = marketplace_contract_json['abi']
                self.marketplace_contract_bytecode = marketplace_contract_json['bytecode']

            # Royalty Contract
            with open('contracts/ArtChainRoyalty.json', 'r') as f:
                royalty_contract_json = json.load(f)
                self.royalty_contract_abi = royalty_contract_json['abi']
                self.royalty_contract_bytecode = royalty_contract_json['bytecode']

            logger.info("Successfully loaded contract definitions")
        except Exception as e:
            logger.error(f"Error loading contract definitions: {e}")

            # Define minimal ABIs for fallback
            self.nft_contract_abi = [
                {
                    "inputs": [
                        {"internalType": "address", "name": "to", "type": "address"},
                        {"internalType": "string", "name": "uri", "type": "string"}
                    ],
                    "name": "mint",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                    "name": "tokenURI",
                    "outputs": [{"internalType": "string", "name": "", "type": "string"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

            self.marketplace_contract_abi = [
                {
                    "inputs": [
                        {"internalType": "address", "name": "nftContract", "type": "address"},
                        {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
                        {"internalType": "uint256", "name": "price", "type": "uint256"}
                    ],
                    "name": "listItem",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "address", "name": "nftContract", "type": "address"},
                        {"internalType": "uint256", "name": "tokenId", "type": "uint256"}
                    ],
                    "name": "buyItem",
                    "outputs": [],
                    "stateMutability": "payable",
                    "type": "function"
                }
            ]

            self.royalty_contract_abi = [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
                        {"internalType": "address[]", "name": "contributors", "type": "address[]"},
                        {"internalType": "uint256[]", "name": "shares", "type": "uint256[]"}
                    ],
                    "name": "setRoyaltyInfo",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                    "name": "getRoyaltyInfo",
                    "outputs": [
                        {"internalType": "address[]", "name": "", "type": "address[]"},
                        {"internalType": "uint256[]", "name": "", "type": "uint256[]"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

            self.nft_contract_bytecode = ""
            self.marketplace_contract_bytecode = ""
            self.royalty_contract_bytecode = ""

    def compile_contract(self, contract_path):
        """Compile Solidity contract"""
        try:
            # Make sure solc is installed
            solcx.install_solc()

            # Compile contract
            compiled_sol = solcx.compile_files(
                [contract_path],
                output_values=['abi', 'bin'],
                optimize=True,
                optimize_runs=200
            )

            # Get contract interface
            contract_id, contract_interface = compiled_sol.popitem()
            abi = contract_interface['abi']
            bytecode = contract_interface['bin']

            return abi, bytecode
        except Exception as e:
            logger.error(f"Error compiling contract: {e}")
            return None, None

    def deploy_contract(self, abi, bytecode, *args):
        """Deploy a contract to the blockchain"""
        if not self.account:
            logger.error("Cannot deploy contract: No private key provided")
            return None

        try:
            # Create contract object
            contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)

            # Build constructor transaction
            constructor_tx = contract.constructor(*args).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price
            })

            # Sign transaction
            signed_tx = self.account.sign_transaction(constructor_tx)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for transaction to be mined
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            contract_address = tx_receipt.contractAddress
            logger.info(f"Contract deployed at address: {contract_address}")

            return contract_address
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            return None

    def load_contract(self, contract_type, address):
        """Load a deployed contract"""
        try:
            if contract_type == 'nft':
                self.nft_contract = self.w3.eth.contract(address=address, abi=self.nft_contract_abi)
                logger.info(f"Loaded NFT contract at address: {address}")
                return self.nft_contract
            elif contract_type == 'marketplace':
                self.marketplace_contract = self.w3.eth.contract(address=address, abi=self.marketplace_contract_abi)
                logger.info(f"Loaded Marketplace contract at address: {address}")
                return self.marketplace_contract
            elif contract_type == 'royalty':
                self.royalty_contract = self.w3.eth.contract(address=address, abi=self.royalty_contract_abi)
                logger.info(f"Loaded Royalty contract at address: {address}")
                return self.royalty_contract
            else:
                logger.error(f"Unknown contract type: {contract_type}")
                return None
        except Exception as e:
            logger.error(f"Error loading contract: {e}")
            return None

    def mint_nft(self, to_address, metadata_uri):
        """Mint a new NFT"""
        if not self.nft_contract or not self.account:
            logger.error("Cannot mint NFT: Contract not loaded or no private key provided")
            return None

        try:
            # Build transaction
            tx = self.nft_contract.functions.mint(to_address, metadata_uri).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price
            })

            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for transaction to be mined
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            # Get token ID from event
            token_id = None
            for log in tx_receipt.logs:
                # Try to decode the event
                try:
                    event = self.nft_contract.events.Transfer().process_log(log)
                    token_id = event['args']['tokenId']
                    break
                except:
                    continue

            if token_id is not None:
                logger.info(f"Minted NFT with token ID: {token_id}")
                return token_id
            else:
                logger.warning("Minted NFT but couldn't determine token ID")
                return "unknown"
        except Exception as e:
            logger.error(f"Error minting NFT: {e}")
            return None

    def list_nft_for_sale(self, nft_contract_address, token_id, price_wei):
        """List an NFT for sale on the marketplace"""
        if not self.marketplace_contract or not self.account:
            logger.error("Cannot list NFT: Contract not loaded or no private key provided")
            return False

        try:
            # Build transaction
            tx = self.marketplace_contract.functions.listItem(
                nft_contract_address, token_id, price_wei
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price
            })

            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for transaction to be mined
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if tx_receipt.status == 1:
                logger.info(f"Listed NFT {token_id} for sale at {price_wei} wei")
                return True
            else:
                logger.error("Transaction failed")
                return False
        except Exception as e:
            logger.error(f"Error listing NFT for sale: {e}")
            return False

    def buy_nft(self, nft_contract_address, token_id, price_wei):
        """Buy an NFT from the marketplace"""
        if not self.marketplace_contract or not self.account:
            logger.error("Cannot buy NFT: Contract not loaded or no private key provided")
            return False

        try:
            # Build transaction
            tx = self.marketplace_contract.functions.buyItem(
                nft_contract_address, token_id
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price,
                'value': price_wei
            })

            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for transaction to be mined
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if tx_receipt.status == 1:
                logger.info(f"Purchased NFT {token_id} for {price_wei} wei")
                return True
            else:
                logger.error("Transaction failed")
                return False
        except Exception as e:
            logger.error(f"Error buying NFT: {e}")
            return False

    def set_royalty_info(self, token_id, contributors, shares):
        """Set royalty information for an NFT"""
        if not self.royalty_contract or not self.account:
            logger.error("Cannot set royalty info: Contract not loaded or no private key provided")
            return False

        try:
            # Build transaction
            tx = self.royalty_contract.functions.setRoyaltyInfo(
                token_id, contributors, shares
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price
            })

            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for transaction to be mined
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if tx_receipt.status == 1:
                logger.info(f"Set royalty info for NFT {token_id}")
                return True
            else:
                logger.error("Transaction failed")
                return False
        except Exception as e:
            logger.error(f"Error setting royalty info: {e}")
            return False

    def get_royalty_info(self, token_id):
        """Get royalty information for an NFT"""
        if not self.royalty_contract:
            logger.error("Cannot get royalty info: Contract not loaded")
            return None, None

        try:
            # Call contract method
            contributors, shares = self.royalty_contract.functions.getRoyaltyInfo(token_id).call()

            logger.info(f"Retrieved royalty info for NFT {token_id}")
            return contributors, shares
        except Exception as e:
            logger.error(f"Error getting royalty info: {e}")
            return None, None

    def create_metadata(self, name, description, image_url, attributes=None):
        """Create metadata for an NFT"""
        metadata = {
            "name": name,
            "description": description,
            "image": image_url,
        }

        if attributes:
            metadata["attributes"] = attributes

        return metadata

    def upload_to_ipfs(self, file_path=None, metadata=None):
        """Upload file or metadata to IPFS and return URI"""
        # Note: This is a placeholder function. In a real implementation,
        # you would use a service like Pinata, Infura, or run an IPFS node.

        if file_path:
            # Mock IPFS hash for file
            file_hash = f"Qm{hashlib.sha256(file_path.encode()).hexdigest()[:44]}"
            ipfs_uri = f"ipfs://{file_hash}"
            logger.info(f"Uploaded file to IPFS: {ipfs_uri}")
            return ipfs_uri

        if metadata:
            # Mock IPFS hash for metadata
            metadata_str = json.dumps(metadata)
            metadata_hash = f"Qm{hashlib.sha256(metadata_str.encode()).hexdigest()[:44]}"
            ipfs_uri = f"ipfs://{metadata_hash}"
            logger.info(f"Uploaded metadata to IPFS: {ipfs_uri}")
            return ipfs_uri

        return None


# API Routes

@app.route('/api/user/register', methods=['POST'])
def register_user():
    """Register a new user"""
    try:
        data = request.json

        # Validate required fields
        if not all(key in data for key in ['username', 'email', 'password']):
            return jsonify({"error": "Missing required fields"}), 400

        # Check if username or email already exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({"error": "Username already exists"}), 400

        if User.query.filter_by(email=data['email']).first():
            return jsonify({"error": "Email already exists"}), 400

        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            is_artist=data.get('is_artist', False),
            wallet_address=data.get('wallet_address', None),
            bio=data.get('bio', None)
        )
        user.set_password(data['password'])

        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User registered successfully", "user_id": user.id}), 201
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.json

        # Validate required fields
        if not all(key in data for key in ['username', 'password']):
            return jsonify({"error": "Missing username or password"}), 400

        # Find user
        user = User.query.filter_by(username=data['username']).first()

        # Check password
        if user and user.check_password(data['password']):
            login_user(user)
            return jsonify({"message": "Login successful", "user": user.to_dict()}), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/logout', methods=['POST'])
@login_required
def logout():
    """Logout user"""
    try:
        logout_user()
        return jsonify({"message": "Logout successful"}), 200
    except Exception as e:
        logger.error(f"Error logging out: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_profile():
    """Get user profile"""
    try:
        return jsonify({"user": current_user.to_dict()}), 200
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        data = request.json

        # Update fields
        if 'username' in data and data['username'] != current_user.username:
            if User.query.filter_by(username=data['username']).first():
                return jsonify({"error": "Username already exists"}), 400
            current_user.username = data['username']

        if 'email' in data and data['email'] != current_user.email:
            if User.query.filter_by(email=data['email']).first():
                return jsonify({"error": "Email already exists"}), 400
            current_user.email = data['email']

        if 'bio' in data:
            current_user.bio = data['bio']

        if 'wallet_address' in data:
            current_user.wallet_address = data['wallet_address']

        if 'is_artist' in data:
            current_user.is_artist = data['is_artist']

        if 'password' in data:
            current_user.set_password(data['password'])

        db.session.commit()

        return jsonify({"message": "Profile updated successfully", "user": current_user.to_dict()}), 200
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/profile_image', methods=['POST'])
@login_required
def upload_profile_image():
    """Upload profile image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        if file and allowed_file(file.filename):
            file_path, _ = save_uploaded_file(file)

            # Update user profile image
            current_user.profile_image = file_path
            db.session.commit()

            return jsonify({"message": "Profile image uploaded successfully", "image_path": file_path}), 200
        else:
            return jsonify({"error": "Invalid file format"}), 400
    except Exception as e:
        logger.error(f"Error uploading profile image: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/artwork/upload', methods=['POST'])
@login_required
def upload_artwork():
    """Upload a new artwork"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Get form data
        title = request.form.get('title', 'Untitled')
        description = request.form.get('description', '')
        tags = request.form.get('tags', '')

        if file and allowed_file(file.filename):
            file_path, content_hash = save_uploaded_file(file)

            # Create new artwork
            artwork = Artwork(
                title=title,
                description=description,
                image_path=file_path,
                artist_id=current_user.id,
                tags=tags,
                content_hash=content_hash
            )

            db.session.add(artwork)
            db.session.commit()

            return jsonify({
                "message": "Artwork uploaded successfully", 
                "artwork": artwork.to_dict()
            }), 201
        else:
            return jsonify({"error": "Invalid file format"}), 400
    except Exception as e:
        logger.error(f"Error uploading artwork: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/artwork/<int:artwork_id>', methods=['GET'])
def get_artwork(artwork_id):
    """Get artwork details"""
    try:
        artwork = Artwork.query.get(artwork_id)

        if not artwork:
            return jsonify({"error": "Artwork not found"}), 404

        return jsonify({"artwork": artwork.to_dict()}), 200
    except Exception as e:
        logger.error(f"Error getting artwork: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/artwork/user/<int:user_id>', methods=['GET'])
def get_user_artworks(user_id):
    """Get all artworks by a user"""
    try:
        artworks = Artwork.query.filter_by(artist_id=user_id).all()

        return jsonify({
            "artworks": [artwork.to_dict() for artwork in artworks],
            "count": len(artworks)
        }), 200
    except Exception as e:
        logger.error(f"Error getting user artworks: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/artwork/<int:artwork_id>', methods=['PUT'])
@login_required
def update_artwork(artwork_id):
    """Update artwork details"""
    try:
        artwork = Artwork.query.get(artwork_id)

        if not artwork:
            return jsonify({"error": "Artwork not found"}), 404

        if artwork.artist_id != current_user.id:
            return jsonify({"error": "Not authorized to update this artwork"}), 403

        data = request.json

        # Update fields
        if 'title' in data:
            artwork.title = data['title']

        if 'description' in data:
            artwork.description = data['description']

        if 'tags' in data:
            artwork.tags = data['tags']

        db.session.commit()

        return jsonify({
            "message": "Artwork updated successfully", 
            "artwork": artwork.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error updating artwork: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/artwork/<int:artwork_id>', methods=['DELETE'])
@login_required
def delete_artwork(artwork_id):
    """Delete an artwork"""
    try:
        artwork = Artwork.query.get(artwork_id)

        if not artwork:
            return jsonify({"error": "Artwork not found"}), 404

        if artwork.artist_id != current_user.id:
            return jsonify({"error": "Not authorized to delete this artwork"}), 403

        # Delete file from disk
        if os.path.exists(artwork.image_path):
            os.remove(artwork.image_path)

        db.session.delete(artwork)
        db.session.commit()

        return jsonify({"message": "Artwork deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting artwork: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/create', methods=['POST'])
@login_required
def create_ai_model():
    """Create a new AI model"""
    try:
        data = request.json

        # Validate required fields
        if not all(key in data for key in ['name', 'model_type']):
            return jsonify({"error": "Missing required fields"}), 400

        # Create new model
        model = AIModel(
            name=data['name'],
            description=data.get('description', ''),
            model_type=data['model_type'],
            owner_id=current_user.id,
            training_params=json.dumps(data.get('training_params', {}))
        )

        db.session.add(model)
        db.session.commit()

        return jsonify({
            "message": "AI model created successfully", 
            "model": model.to_dict()
        }), 201
    except Exception as e:
        logger.error(f"Error creating AI model: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/<int:model_id>/add_training_data', methods=['POST'])
@login_required
def add_training_data(model_id):
    """Add artwork to model training data"""
    try:
        model = AIModel.query.get(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.owner_id != current_user.id:
            return jsonify({"error": "Not authorized to update this model"}), 403

        data = request.json

        if 'artwork_ids' not in data or not isinstance(data['artwork_ids'], list):
            return jsonify({"error": "artwork_ids list is required"}), 400

        # Add artworks to training data
        artworks_added = []
        for artwork_id in data['artwork_ids']:
            artwork = Artwork.query.get(artwork_id)

            if not artwork:
                continue

            # Check if already in training data
            if artwork in model.training_artworks:
                continue

            model.training_artworks.append(artwork)
            artwork.is_used_for_training = True
            artworks_added.append(artwork_id)

        db.session.commit()

        return jsonify({
            "message": f"Added {len(artworks_added)} artworks to training data",
            "artworks_added": artworks_added
        }), 200
    except Exception as e:
        logger.error(f"Error adding training data: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/<int:model_id>/train', methods=['POST'])
@login_required
def train_model(model_id):
    """Start training an AI model"""
    try:
        model = AIModel.query.get(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.owner_id != current_user.id:
            return jsonify({"error": "Not authorized to train this model"}), 403

        if len(model.training_artworks) == 0:
            return jsonify({"error": "No training data available"}), 400

        # Update model status
        model.training_status = 'training'
        db.session.commit()

        # In a real implementation, you would start a background task for training
        # For now, we'll simulate successful training

        # Create model directory if it doesn't exist
        model_dir = os.path.join('models', str(model.id))
        os.makedirs(model_dir, exist_ok=True)

        # Generate model path
        model_path = os.path.join(model_dir, f"{model.model_type.lower()}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt")

        # Update model path and status
        model.model_path = model_path
        model.training_status = 'completed'
        model.last_trained = datetime.datetime.utcnow()
        db.session.commit()

        return jsonify({
            "message": "Model training started",
            "model": model.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error training model: {e}")
        model.training_status = 'failed'
        db.session.commit()
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/<int:model_id>/generate', methods=['POST'])
@login_required
def generate_art(model_id):
    """Generate art using an AI model"""
    try:
        model = AIModel.query.get(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.training_status != 'completed':
            return jsonify({"error": "Model is not ready for generation"}), 400

        data = request.json
        num_images = data.get('num_images', 1)
        seed = data.get('seed', None)

        # Create output directory if it doesn't exist
        output_dir = os.path.join('generated', str(model.id))
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # Generate images based on model type
        image_paths = []

        if model.model_type.lower() == 'gan':
            # Initialize GAN model
            gan_model = GANModel(model_path=model.model_path)

            # Generate images
            output_base = os.path.join(output_dir, f"gan_{timestamp}")
            image_paths = gan_model.generate_image(output_base, num_images=num_images, seed=seed)

        elif model.model_type.lower() == 'diffusion':
            # Initialize Diffusion model
            diffusion_model = DiffusionModel(model_path=model.model_path)

            # Generate images
            output_base = os.path.join(output_dir, f"diffusion_{timestamp}")
            image_paths = diffusion_model.generate_image(output_base, num_images=num_images, seed=seed)

        elif model.model_type.lower() == 'style_transfer':
            # For style transfer, we need content image
            if 'content_image_id' not in data:
                return jsonify({"error": "content_image_id is required for style transfer"}), 400

            # Get content image
            content_artwork = Artwork.query.get(data['content_image_id'])
            if not content_artwork:
                return jsonify({"error": "Content image not found"}), 404

            # Get style image (first training artwork)
            if not model.training_artworks:
                return jsonify({"error": "No style images available"}), 400

            style_artwork = model.training_artworks[0]

            # Initialize Style Transfer model
            style_model = StyleTransferModel()

            # Generate image
            output_path = os.path.join(output_dir, f"style_{timestamp}.png")
            generated_path = style_model.generate_image(
                content_artwork.image_path,
                style_artwork.image_path,
                output_path
            )

            image_paths = [generated_path]

        # Create NFTs for generated images
        nfts = []
        for i, image_path in enumerate(image_paths):
            # Calculate hash for the image
            content_hash = hash_file(image_path)

            # Create NFT
            nft = NFT(
                title=f"{model.name} Generation #{i+1}",
                description=f"AI-generated art using {model.name} model",
                image_path=image_path,
                creator_id=current_user.id,
                ai_model_id=model.id,
                content_hash=content_hash,
                original_artwork_ids=','.join([str(artwork.id) for artwork in model.training_artworks])
            )

            db.session.add(nft)
            db.session.commit()

            nfts.append(nft.to_dict())

        return jsonify({
            "message": f"Generated {len(image_paths)} images",
            "nfts": nfts
        }), 200
    except Exception as e:
        logger.error(f"Error generating art: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/nft/<int:nft_id>/mint', methods=['POST'])
@login_required
def mint_nft_token(nft_id):
    """Mint an NFT on the blockchain"""
    try:
        nft = NFT.query.get(nft_id)

        if not nft:
            return jsonify({"error": "NFT not found"}), 404

        if nft.creator_id != current_user.id:
            return jsonify({"error": "Not authorized to mint this NFT"}), 403

        if nft.status != 'pending':
            return jsonify({"error": f"NFT is already in {nft.status} state"}), 400

        # Initialize blockchain manager
        blockchain_manager = SmartContractManager()

        # Load NFT contract
        if not blockchain_manager.nft_contract and CONTRACT_ADDRESS:
            blockchain_manager.load_contract('nft', CONTRACT_ADDRESS)

        # Upload image to IPFS
        image_uri = blockchain_manager.upload_to_ipfs(file_path=nft.image_path)

        # Create metadata
        metadata = blockchain_manager.create_metadata(
            name=nft.title,
            description=nft.description,
            image_url=image_uri,
            attributes=[
                {"trait_type": "Artist", "value": current_user.username},
                {"trait_type": "AI Model", "value": AIModel.query.get(nft.ai_model_id).name if nft.ai_model_id else "None"},
                {"trait_type": "Creation Date", "value": nft.created_at.strftime('%Y-%m-%d')}
            ]
        )

        # Upload metadata to IPFS
        metadata_uri = blockchain_manager.upload_to_ipfs(metadata=metadata)

        # Mint NFT
        token_id = blockchain_manager.mint_nft(current_user.wallet_address, metadata_uri)

        if token_id:
            # Update NFT status
            nft.status = 'minted'
            nft.token_id = str(token_id)
            nft.contract_address = CONTRACT_ADDRESS
            nft.metadata_uri = metadata_uri

            db.session.commit()

            # Set up royalties if it's a collaborative work
            if nft.original_artwork_ids:
                artwork_ids = [int(id) for id in nft.original_artwork_ids.split(',')]
                original_artists = set()

                for artwork_id in artwork_ids:
                    artwork = Artwork.query.get(artwork_id)
                    if artwork:
                        original_artists.add(artwork.artist_id)

                # If multiple artists, set up royalty distribution
                if len(original_artists) > 1:
                    contributors = [User.query.get(artist_id).wallet_address for artist_id in original_artists]
                    shares = [100 // len(contributors)] * len(contributors)

                    # Adjust last share to make sure they sum to 100
                    shares[-1] = 100 - sum(shares[:-1])

                    blockchain_manager.set_royalty_info(token_id, contributors, shares)

            return jsonify({
                "message": "NFT minted successfully",
                "nft": nft.to_dict()
            }), 200
        else:
            return jsonify({"error": "Failed to mint NFT"}), 500
    except Exception as e:
        logger.error(f"Error minting NFT: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/nft/<int:nft_id>/list', methods=['POST'])
@login_required
def list_nft(nft_id):
    """List an NFT for sale"""
    try:
        nft = NFT.query.get(nft_id)

        if not nft:
            return jsonify({"error": "NFT not found"}), 404

        if nft.creator_id != current_user.id:
            return jsonify({"error": "Not authorized to list this NFT"}), 403

        if nft.status != 'minted':
            return jsonify({"error": f"NFT is in {nft.status} state, must be minted"}), 400

        data = request.json

        if 'price' not in data:
            return jsonify({"error": "Price is required"}), 400

        price = float(data['price'])
        currency = data.get('currency', 'MATIC')

        # Convert price to wei
        price_wei = int(price * 10**18)  # Assuming 18 decimals for MATIC

        # Initialize blockchain manager
        blockchain_manager = SmartContractManager()

        # Load marketplace contract
        marketplace_address = data.get('marketplace_address')
        if marketplace_address:
            blockchain_manager.load_contract('marketplace', marketplace_address)

        # List NFT for sale
        success = blockchain_manager.list_nft_for_sale(nft.contract_address, nft.token_id, price_wei)

        if success:
            # Update NFT status and price
            nft.status = 'listed'
            nft.price = price
            nft.currency = currency

            db.session.commit()

            return jsonify({
                "message": "NFT listed for sale",
                "nft": nft.to_dict()
            }), 200
        else:
            return jsonify({"error": "Failed to list NFT for sale"}), 500
    except Exception as e:
        logger.error(f"Error listing NFT: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/nft/<int:nft_id>/buy', methods=['POST'])
@login_required
def buy_nft(nft_id):
    """Buy an NFT"""
    try:
        nft = NFT.query.get(nft_id)

        if not nft:
            return jsonify({"error": "NFT not found"}), 404

        if nft.status != 'listed':
            return jsonify({"error": f"NFT is not listed for sale"}), 400

        if nft.creator_id == current_user.id:
            return jsonify({"error": "You cannot buy your own NFT"}), 400

        # Convert price to wei
        price_wei = int(nft.price * 10**18)  # Assuming 18 decimals for MATIC

        # Initialize blockchain manager
        blockchain_manager = SmartContractManager()

        # Load marketplace contract
        marketplace_address = request.json.get('marketplace_address')
        if marketplace_address:
            blockchain_manager.load_contract('marketplace', marketplace_address)

        # Buy NFT
        success = blockchain_manager.buy_nft(nft.contract_address, nft.token_id, price_wei)

        if success:
            # Create transaction record
            transaction = Transaction(
                nft_id=nft.id,
                seller_id=nft.creator_id,
                buyer_id=current_user.id,
                amount=nft.price,
                currency=nft.currency,
                status='completed'
            )

            # Update NFT status
            nft.status = 'sold'

            db.session.add(transaction)
            db.session.commit()

            return jsonify({
                "message": "NFT purchased successfully",
                "transaction": transaction.to_dict()
            }), 200
        else:
            return jsonify({"error": "Failed to purchase NFT"}), 500
    except Exception as e:
        logger.error(f"Error buying NFT: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/nft/marketplace', methods=['GET'])
def get_marketplace_nfts():
    """Get all NFTs listed for sale"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)

        # Query NFTs listed for sale
        query = NFT.query.filter_by(status='listed')

        # Apply filters if provided
        if 'creator_id' in request.args:
            query = query.filter_by(creator_id=request.args.get('creator_id', type=int))

        if 'min_price' in request.args:
            query = query.filter(NFT.price >= request.args.get('min_price', type=float))

        if 'max_price' in request.args:
            query = query.filter(NFT.price <= request.args.get('max_price', type=float))

        # Paginate results
        nfts = query.paginate(page=page, per_page=per_page)

        return jsonify({
            "nfts": [nft.to_dict() for nft in nfts.items],
            "total": nfts.total,
            "pages": nfts.pages,
            "page": nfts.page
        }), 200
    except Exception as e:
        logger.error(f"Error getting marketplace NFTs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/collaboration/create', methods=['POST'])
@login_required
def create_collaboration():
    """Create a new artist collaboration"""
    try:
        data = request.json

        # Validate required fields
        if not all(key in data for key in ['name', 'royalty_distribution']):
            return jsonify({"error": "Missing required fields"}), 400

        # Validate royalty distribution format and sum
        royalty_distribution = data['royalty_distribution']
        if not isinstance(royalty_distribution, dict):
            return jsonify({"error": "royalty_distribution must be a dictionary of user_id: percentage pairs"}), 400

        total_percentage = sum(royalty_distribution.values())
        if total_percentage != 100:
            return jsonify({"error": "Royalty percentages must sum to 100"}), 400

        # Create new collaboration
        collaboration = Collaboration(
            name=data['name'],
            description=data.get('description', ''),
            creator_id=current_user.id,
            royalty_distribution=json.dumps(royalty_distribution),
            status='draft'
        )

        db.session.add(collaboration)
        db.session.commit()

        return jsonify({
            "message": "Collaboration created successfully",
            "collaboration": collaboration.to_dict()
        }), 201
    except Exception as e:
        logger.error(f"Error creating collaboration: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/collaboration/<int:collaboration_id>/add_model', methods=['POST'])
@login_required
def add_model_to_collaboration(collaboration_id):
    """Add an AI model to a collaboration"""
    try:
        collaboration = Collaboration.query.get(collaboration_id)

        if not collaboration:
            return jsonify({"error": "Collaboration not found"}), 404

        # Check if user is part of the collaboration
        royalty_distribution = json.loads(collaboration.royalty_distribution)
        if str(current_user.id) not in royalty_distribution:
            return jsonify({"error": "Not authorized to add models to this collaboration"}), 403

        data = request.json

        if 'model_id' not in data:
            return jsonify({"error": "model_id is required"}), 400

        model = AIModel.query.get(data['model_id'])

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.owner_id != current_user.id:
            return jsonify({"error": "You can only add your own models to a collaboration"}), 403

        # Add model to collaboration
        if model not in collaboration.models:
            collaboration.models.append(model)

        db.session.commit()

        return jsonify({
            "message": "Model added to collaboration",
            "model": model.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error adding model to collaboration: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/collaboration/<int:collaboration_id>/generate', methods=['POST'])
@login_required
def generate_collaborative_art(collaboration_id):
    """Generate art using models from a collaboration"""
    try:
        collaboration = Collaboration.query.get(collaboration_id)

        if not collaboration:
            return jsonify({"error": "Collaboration not found"}), 404

        # Check if user is part of the collaboration
        royalty_distribution = json.loads(collaboration.royalty_distribution)
        if str(current_user.id) not in royalty_distribution:
            return jsonify({"error": "Not authorized to generate art with this collaboration"}), 403

        if not collaboration.models:
            return jsonify({"error": "No models available in this collaboration"}), 400

        data = request.json

        # Create output directory if it doesn't exist
        output_dir = os.path.join('generated', f"collab_{collaboration_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # For simplicity, we'll use the first model to generate art
        # In a real implementation, you might combine multiple models
        model = collaboration.models[0]

        # Generate art based on model type
        if model.model_type.lower() == 'gan':
            # Initialize GAN model
            gan_model = GANModel(model_path=model.model_path)

            # Generate image
            output_base = os.path.join(output_dir, f"collab_gan_{timestamp}")
            image_path = gan_model.generate_image(output_base, num_images=1)[0]

        elif model.model_type.lower() == 'diffusion':
            # Initialize Diffusion model
            diffusion_model = DiffusionModel(model_path=model.model_path)

            # Generate image
            output_base = os.path.join(output_dir, f"collab_diffusion_{timestamp}")
            image_path = diffusion_model.generate_image(output_base, num_images=1)[0]

        else:
            return jsonify({"error": f"Unsupported model type: {model.model_type}"}), 400

        # Calculate hash for the image
        content_hash = hash_file(image_path)

        # Create NFT
        nft = NFT(
            title=f"{collaboration.name} Generation",
            description=f"Collaborative AI-generated art",
            image_path=image_path,
            creator_id=current_user.id,
            ai_model_id=model.id,
            content_hash=content_hash,
            status='pending'
        )

        db.session.add(nft)
        db.session.commit()

        return jsonify({
            "message": "Generated collaborative art",
            "nft": nft.to_dict(),
            "royalty_distribution": royalty_distribution
        }), 200
    except Exception as e:
        logger.error(f"Error generating collaborative art: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/user/<int:user_id>', methods=['GET'])
def get_user_stats(user_id):
    """Get statistics for a user"""
    try:
        user = User.query.get(user_id)

        if not user:
            return jsonify({"error": "User not found"}), 404

        # Count artworks
        artwork_count = Artwork.query.filter_by(artist_id=user_id).count()

        # Count AI models
        model_count = AIModel.query.filter_by(owner_id=user_id).count()

        # Count NFTs
        nft_count = NFT.query.filter_by(creator_id=user_id).count()

        # Count transactions as seller
        sales_count = Transaction.query.filter_by(seller_id=user_id, status='completed').count()

        # Calculate total sales
        sales = Transaction.query.filter_by(seller_id=user_id, status='completed').all()
        total_sales = sum(transaction.amount for transaction in sales)

        # Count transactions as buyer
        purchases_count = Transaction.query.filter_by(buyer_id=user_id, status='completed').count()

        return jsonify({
            "user": user.to_dict(),
            "stats": {
                "artwork_count": artwork_count,
                "model_count": model_count,
                "nft_count": nft_count,
                "sales_count": sales_count,
                "total_sales": total_sales,
                "purchases_count": purchases_count
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/platform', methods=['GET'])
def get_platform_stats():
    """Get platform-wide statistics"""
    try:
        # Count users
        user_count = User.query.count()
        artist_count = User.query.filter_by(is_artist=True).count()

        # Count artworks
        artwork_count = Artwork.query.count()

        # Count AI models
        model_count = AIModel.query.count()

        # Count NFTs
        nft_count = NFT.query.count()
        minted_nft_count = NFT.query.filter(NFT.status != 'pending').count()

        # Count transactions
        transaction_count = Transaction.query.filter_by(status='completed').count()

        # Calculate total sales volume
        transactions = Transaction.query.filter_by(status='completed').all()
        total_volume = sum(transaction.amount for transaction in transactions)

        return jsonify({
            "stats": {
                "user_count": user_count,
                "artist_count": artist_count,
                "artwork_count": artwork_count,
                "model_count": model_count,
                "nft_count": nft_count,
                "minted_nft_count": minted_nft_count,
                "transaction_count": transaction_count,
                "total_volume": total_volume
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        return jsonify({"error": str(e)}), 500


# Web Routes

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')


@app.route('/login')
def login_page():
    """Render login page"""
    return render_template('login.html')


@app.route('/register')
def register_page():
    """Render registration page"""
    return render_template('register.html')


@app.route('/profile')
@login_required
def profile_page():
    """Render user profile page"""
    return render_template('profile.html')


@app.route('/artwork')
def artwork_page():
    """Render artwork browser page"""
    return render_template('artwork.html')


@app.route('/models')
@login_required
def models_page():
    """Render AI models page"""
    return render_template('models.html')


@app.route('/marketplace')
def marketplace_page():
    """Render NFT marketplace page"""
    return render_template('marketplace.html')


@app.route('/collaborations')
@login_required
def collaborations_page():
    """Render collaborations page"""
    return render_template('collaborations.html')


# Main entry point

if __name__ == '__main__':
    with app.app_context():
        # Create all database tables
        db.create_all()

    # Start the Flask application
    app.run(debug=True)