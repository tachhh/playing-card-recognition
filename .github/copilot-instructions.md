# Copilot Instructions for Playing Card Recognition

## Project Overview

This repository contains a playing card recognition system that uses computer vision and machine learning techniques to identify and classify playing cards from images.

## Development Guidelines

### Code Style and Quality

- Write clean, readable, and well-documented code
- Follow PEP 8 style guidelines for Python code (if Python is used)
- Use meaningful variable and function names that clearly describe their purpose
- Add docstrings to all functions and classes explaining their purpose, parameters, and return values
- Keep functions small and focused on a single responsibility

### Computer Vision Best Practices

- When working with image processing, always validate input images for proper format and dimensions
- Use appropriate preprocessing techniques (grayscale conversion, normalization, resizing) before model inference
- Document any image size requirements or constraints for the card recognition models
- Consider edge cases like rotated cards, partially visible cards, or multiple cards in a single image

### Machine Learning Considerations

- If adding or modifying ML models:
  - Document model architecture, input/output shapes, and preprocessing requirements
  - Include information about training data and model performance metrics
  - Provide clear instructions for model training, evaluation, and inference
  - Version control model files and track model performance over time

### Testing

- Write unit tests for individual components (preprocessing, model inference, utilities)
- Include integration tests for end-to-end card recognition workflows
- Test with various card types, orientations, and lighting conditions
- Validate model performance on a diverse test dataset

### Data Management

- Keep training/test datasets organized and documented
- Use version control for small datasets or reference images
- For large datasets, document how to obtain or generate them
- Include sample images in the repository for testing and demonstration

### Performance

- Optimize image processing pipelines for efficiency
- Consider real-time processing requirements if applicable
- Profile code to identify and address performance bottlenecks
- Document inference time and resource requirements

## Project Structure

When adding new features or files, follow these organizational principles:

- `/data` - Training and test datasets, sample images
- `/models` - Trained model files and architectures
- `/src` - Source code for card recognition algorithms
- `/tests` - Unit and integration tests
- `/notebooks` - Jupyter notebooks for experimentation and visualization
- `/docs` - Additional documentation and examples

## Dependencies

- When adding new dependencies, add them to requirements.txt or package.json
- Pin dependency versions to ensure reproducibility
- Document any system-level dependencies (OpenCV, CUDA, etc.)

## Contributing

- Make small, focused commits with clear commit messages
- Test all changes thoroughly before committing
- Update documentation when adding new features
- Consider backward compatibility when making changes
