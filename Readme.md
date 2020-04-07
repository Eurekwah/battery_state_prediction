---
noteId: "487744f078d311eab357ede431cef9a9"
tags: []

---

# Battery state of charge prediction based on machine learning algorithm


> Version 1.0 : Re-write the Electric vehicles and batteries simulation model, build a CNN-LSTM model for SOC prediction, the SOH is also considered.

# Platform 
- Python: 3.x
- Pytorch: 0.4+
- Matlab: 2019

# Electirc Vehicles Model

## Howto

- Determine the weight, windward area, internal drive efficiency and the numbers of batteries of the simulation objects
- Fill them into the corresponding position of the model
- The speed and slope is also needed

## Introduction 

### Structure
- Operating input module
- Automotive speed control module 
- Torque conversion module 
- Vehicle mechanics module 
- Power conversion module 
- Thermodynamics module
