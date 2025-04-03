# MyEmma Frontend

This is a Streamlit-based frontend for the MyEmma end-to-end autonomous driving framework.

## Features

- Dataset selection and configuration
- Qianwen API integration
- Real-time visualization of camera views and trajectory predictions
- Display of AI-generated scene descriptions, object detections, and intent predictions
- Interactive navigation through scene frames

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the NuScenes dataset is available in the specified path.

## Usage

1. Run the Streamlit app:

```bash
cd frontend
streamlit run app.py
```

2. Configure the settings in the sidebar:
   - Set the NuScenes dataset path
   - Select the dataset version
   - Enter your Qianwen API key
   - Click "Run Analysis" to process the data

3. Once processing is complete, you can:
   - Select different result folders
   - Choose specific scenes
   - Navigate through frames
   - Click on images to enlarge them

## Visualization Explained

The app displays:

- Six camera views (front, front-left, front-right, back, back-left, back-right)
- Trajectory prediction in the center
- AI-generated descriptions on the right
- Navigation controls on the left

Click on any image to view it in full size in the sidebar.

## Integration with MyEmma

This frontend integrates with the MyEmma framework by:

1. Calling the backend processing code
2. Loading and displaying the generated results
3. Providing an interactive interface for exploring the data

## Customization

You can modify the app.py file to:

- Change the layout
- Add more visualization options
- Integrate with other models or data sources 