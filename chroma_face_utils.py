import chromadb
import numpy as np
import json
import cv2
import dlib
import os
from typing import List, Dict, Optional, Tuple

class ChromaFaceSearch:
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize ChromaDB client and collection for face embeddings"""
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection for face embeddings
        try:
            self.collection = self.client.get_collection("face_embeddings")
        except:
            self.collection = self.client.create_collection(
                name="face_embeddings",
                metadata={"description": "Face embeddings for similarity search"}
            )
        
        # Initialize face detection and recognition models
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    
    def extract_face_encodings(self, image_path: str) -> List[np.ndarray]:
        """Extract face encodings from an image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Convert to RGB (dlib uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.face_detector(rgb_image)
            
            encodings = []
            for face in faces:
                # Get facial landmarks
                landmarks = self.shape_predictor(rgb_image, face)
                
                # Get face encoding
                encoding = self.face_rec_model.compute_face_descriptor(rgb_image, landmarks)
                encodings.append(np.array(encoding))
            
            return encodings
            
        except Exception as e:
            print(f"Error extracting face encodings: {e}")
            return []
    
    def add_image(self, image_url: str, image_path: str, category: str = None) -> bool:
        """Add face encodings from an image to ChromaDB"""
        try:
            # Extract face encodings
            encodings = self.extract_face_encodings(image_path)
            
            if not encodings:
                return False
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            
            for i, encoding in enumerate(encodings):
                face_id = f"{image_url}_{i}"
                ids.append(face_id)
                embeddings.append(encoding.tolist())
                
                metadata = {
                    "url": image_url,
                    "face_index": i,
                    "total_faces": len(encodings)
                }
                if category:
                    metadata["category"] = category
                
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding image to ChromaDB: {e}")
            return False
    
    def search_similar_faces(self, query_image_path: str, category: str = None, 
                           n_results: int = 10, threshold: float = 0.4) -> List[Dict]:
        """Search for similar faces in ChromaDB"""
        try:
            # Extract face encodings from query image
            query_encodings = self.extract_face_encodings(query_image_path)
            
            if not query_encodings:
                return []
            
            results = []
            
            for query_encoding in query_encodings:
                # Prepare where clause for category filtering
                where_clause = None
                if category:
                    where_clause = {"category": category}
                
                # Search in ChromaDB
                search_results = self.collection.query(
                    query_embeddings=[query_encoding.tolist()],
                    n_results=n_results,
                    where=where_clause
                )
                
                # Process results
                for i, (distance, metadata, doc_id) in enumerate(zip(
                    search_results['distances'][0],
                    search_results['metadatas'][0],
                    search_results['ids'][0]
                )):
                    # ChromaDB uses cosine distance, convert to similarity
                    similarity = 1 - distance
                    
                    if similarity >= (1 - threshold):  # threshold conversion
                        results.append({
                            'url': metadata['url'],
                            'similarity': similarity,
                            'distance': distance,
                            'face_index': metadata.get('face_index', 0),
                            'category': metadata.get('category', None)
                        })
            
            # Remove duplicates and sort by similarity
            unique_urls = {}
            for result in results:
                url = result['url']
                if url not in unique_urls or result['similarity'] > unique_urls[url]['similarity']:
                    unique_urls[url] = result
            
            return sorted(unique_urls.values(), key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            print(f"Error searching similar faces: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the face database"""
        try:
            count = self.collection.count()
            return {
                "total_face_embeddings": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_face_embeddings": 0}
    
    def delete_by_url(self, image_url: str) -> bool:
        """Delete all face embeddings for a specific URL"""
        try:
            # Get all IDs for this URL
            results = self.collection.query(
                query_embeddings=[[0] * 128],  # dummy query
                where={"url": image_url},
                n_results=1000  # large number to get all
            )
            
            if results['ids'] and results['ids'][0]:
                self.collection.delete(ids=results['ids'][0])
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting by URL: {e}")
            return False
    
    def list_categories(self) -> List[str]:
        """List all available categories"""
        try:
            # This is a workaround since ChromaDB doesn't have direct category listing
            # We'll get a sample of records and extract unique categories
            results = self.collection.query(
                query_embeddings=[[0] * 128],  # dummy query
                n_results=1000
            )
            
            categories = set()
            if results['metadatas']:
                for metadata_list in results['metadatas']:
                    for metadata in metadata_list:
                        if 'category' in metadata and metadata['category']:
                            categories.add(metadata['category'])
            
            return sorted(list(categories))
            
        except Exception as e:
            print(f"Error listing categories: {e}")
            return []
    
    def get_images_by_category(self, category: str) -> List[Dict]:
        """Get all unique images for a specific category"""
        try:
            # Get all embeddings for this category
            results = self.collection.query(
                query_embeddings=[[0] * 128],  # dummy query
                where={"category": category},
                n_results=1000
            )
            
            # Extract unique URLs with their metadata
            unique_images = {}
            if results['metadatas']:
                for i, metadata_list in enumerate(results['metadatas']):
                    for j, metadata in enumerate(metadata_list):
                        url = metadata.get('url', '')
                        if url and url not in unique_images:
                            unique_images[url] = {
                                'id': results['ids'][i][j] if results['ids'] else '',
                                'url': url,
                                'category': category
                            }
            
            return list(unique_images.values())
            
        except Exception as e:
            print(f"Error getting images by category: {e}")
            return []