""" import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Dict, Union, Optional

# Sample coffee history (can be expanded)
coffee_history = [
    {"coffee_type": "espresso", "size": "small", "strength": "normal", "price": 2.5},
    {"coffee_type": "latte", "size": "medium", "strength": "normal", "price": 3.5},
    {"coffee_type": "cappuccino", "size": "large", "strength": "strong", "price": 4.0},
    {"coffee_type": "mocha", "size": "small", "strength": "weak", "price": 3.0},
    {"coffee_type": "espresso", "size": "medium", "strength": "strong", "price": 2.8},
    {"coffee_type": "latte", "size": "large", "strength": "normal", "price": 3.7}
]

# Coffee recipes
coffee_recipes = {
    "espresso": {"small": {"water": 50, "milk": 0, "coffee beans": 10},
                 "medium": {"water": 70, "milk": 0, "coffee beans": 15},
                 "large": {"water": 100, "milk": 0, "coffee beans": 20}},
    "latte": {"small": {"water": 50, "milk": 150, "coffee beans": 15},
              "medium": {"water": 70, "milk": 200, "coffee beans": 20},
              "large": {"water": 100, "milk": 250, "coffee beans": 25}},
    "cappuccino": {"small": {"water": 50, "milk": 100, "coffee beans": 15},
                   "medium": {"water": 70, "milk": 150, "coffee beans": 20},
                   "large": {"water": 100, "milk": 200, "coffee beans": 25}},
    "mocha": {"small": {"water": 50, "milk": 100, "coffee beans": 10},
              "medium": {"water": 70, "milk": 150, "coffee beans": 15},
              "large": {"water": 100, "milk": 200, "coffee beans": 20}},
}

# Initial resources
initial_resources = {"water": 1000, "milk": 1000, "coffee beans": 500}

# Function to display resources
def display_resources(resources: Dict[str, float]) -> str:
    return f"Water: {resources['water']}ml, Milk: {resources['milk']}ml, Coffee Beans: {resources['coffee beans']}g"

# Function to refill resources
def refill_resources(resources: Dict[str, float], refill_amount: Dict[str, float]) -> Dict[str, float]:
    for key in refill_amount:
        resources[key] += refill_amount[key]
    return resources

# Function to add custom coffee recipe
def add_custom_coffee_recipe(coffee_type: str, recipe: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    coffee_recipes[coffee_type] = recipe
    return coffee_recipes

# Function to display coffee history (to be expanded as per your project)
def display_history():
    return coffee_history

# Function to make coffee
def coffee_machine(resources: Dict[str, float], coffee_choice: str, size: str, strength: str, price: float) -> Dict[str, float]:
    if coffee_choice in coffee_recipes and size in coffee_recipes[coffee_choice]:
        ingredients = coffee_recipes[coffee_choice][size]
        if resources["water"] >= ingredients["water"] and resources["milk"] >= ingredients["milk"] and resources["coffee beans"] >= ingredients["coffee beans"]:
            resources["water"] -= ingredients["water"]
            resources["milk"] -= ingredients["milk"]
            resources["coffee beans"] -= ingredients["coffee beans"]
            print(f"Making {coffee_choice} ({size}, {strength}) - Price: ${price:.2f}")
        else:
            print("Not enough resources!")
    else:
        print("Invalid coffee choice or size.")
    return resources

# Convert coffee history to a DataFrame
def create_dataset(coffee_history: List[Dict[str, Union[str, float]]]) -> pd.DataFrame:
    df = pd.DataFrame(coffee_history)
    return df

# Encode categorical features (coffee_type, size, strength)
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df["coffee_type_encoded"] = le.fit_transform(df["coffee_type"])
    df["size_encoded"] = le.fit_transform(df["size"])
    df["strength_encoded"] = le.fit_transform(df["strength"])
    return df

# Prepare features and target for ML
def prepare_data(df: pd.DataFrame):
    X = df[["size_encoded", "strength_encoded"]]
    y = df["coffee_type_encoded"]
    return X, y

# Train the ML model
def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a decision tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Predict coffee choice based on size and strength
def predict_coffee(model, size: str, strength: str) -> str:
    le = LabelEncoder()
    size_encoded = le.fit_transform([size])[0]
    strength_encoded = le.fit_transform([strength])[0]
    
    # Predict the coffee type
    prediction = model.predict([[size_encoded, strength_encoded]])
    coffee_type = le.inverse_transform(prediction)[0]
    return coffee_type

# Coffee machine workflow with ML integration
def coffee_machine_with_ml(resources: Dict[str, float], coffee_choice: Optional[str] = None, size: Optional[str] = None, strength: Optional[str] = None, price: Optional[float] = None) -> Dict[str, float]:
    print("Welcome to the Coffee Machine with ML!")
    print("Current resources:")
    print(display_resources(resources))

    # If no coffee choice is provided, use ML to predict
    if not coffee_choice:
        # Create dataset from coffee history
        df = create_dataset(coffee_history)
        if not df.empty:
            df = encode_features(df)
            X, y = prepare_data(df)
            model = train_model(X, y)
            
            # Predict coffee choice based on user input for size and strength
            if not size:
                size = input("Choose size (small/medium/large): ").strip().lower()
            if not strength:
                strength = input("Choose strength (weak/normal/strong): ").strip().lower()
            
            coffee_choice = predict_coffee(model, size, strength)
            print(f"\nBased on your preferences, we recommend: {coffee_choice} ({size}, {strength})")
        else:
            print("No historical data available. Please choose a coffee manually.")
            coffee_choice = input("Enter your coffee choice (espresso, latte, cappuccino, mocha): ").strip().lower()
            size = input("Choose size (small/medium/large): ").strip().lower()
            strength = input("Choose strength (weak/normal/strong): ").strip().lower()
    
    # Make the coffee
    if coffee_choice in coffee_recipes and size in coffee_recipes[coffee_choice] and strength in ["weak", "normal", "strong"]:
        price = coffee_prices.get(coffee_choice, {}).get(size, 0.0)
        resources = coffee_machine(resources, coffee_choice, size, strength, price)
    else:
        print("Invalid coffee choice, size, or strength. Please select from the available options.")

    # Show coffee making history
    print("\nCoffee Making History:")
    print(display_history())

    # Refill resources if needed
    refill_choice = input("\nWould you like to refill resources (yes/no)? ").strip().lower()
    if refill_choice == "yes":
        refill_amount = {"water": 1000, "milk": 1000, "coffee beans": 500}
        resources = refill_resources(resources, refill_amount)
        print("\nResources after refill:")
        print(display_resources(resources))

    return resources

# Example usage
if __name__ == "__main__":
    resources = initial_resources
    
    # Add custom coffee recipe
    print("Add custom coffee recipe:")
    coffee_recipes = add_custom_coffee_recipe("mocha", {
        "small": {"water": 150, "milk": 100, "coffee beans": 20},
        "medium": {"water": 200, "milk": 150, "coffee beans": 25},
        "large": {"water": 250, "milk": 200, "coffee beans": 30}
    })
    print(f"Updated Coffee Recipes: {coffee_recipes}")
    
    # Let the user make a coffee (with ML recommendation)
    resources = coffee_machine_with_ml(resources)



 """
 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Dict, Union, Optional

# Sample coffee history (can be expanded)
coffee_history = [
    {"coffee_type": "espresso", "size": "small", "strength": "normal", "price": 2.5},
    {"coffee_type": "latte", "size": "medium", "strength": "normal", "price": 3.5},
    {"coffee_type": "cappuccino", "size": "large", "strength": "strong", "price": 4.0},
    {"coffee_type": "mocha", "size": "small", "strength": "weak", "price": 3.0},
    {"coffee_type": "espresso", "size": "medium", "strength": "strong", "price": 2.8},
    {"coffee_type": "latte", "size": "large", "strength": "normal", "price": 3.7}
]

# Coffee recipes
coffee_recipes = {
    "espresso": {"small": {"water": 50, "milk": 0, "coffee beans": 10},
                 "medium": {"water": 70, "milk": 0, "coffee beans": 15},
                 "large": {"water": 100, "milk": 0, "coffee beans": 20}},
    "latte": {"small": {"water": 50, "milk": 150, "coffee beans": 15},
              "medium": {"water": 70, "milk": 200, "coffee beans": 20},
              "large": {"water": 100, "milk": 250, "coffee beans": 25}},
    "cappuccino": {"small": {"water": 50, "milk": 100, "coffee beans": 15},
                   "medium": {"water": 70, "milk": 150, "coffee beans": 20},
                   "large": {"water": 100, "milk": 200, "coffee beans": 25}},
    "mocha": {"small": {"water": 50, "milk": 100, "coffee beans": 10},
              "medium": {"water": 70, "milk": 150, "coffee beans": 15},
              "large": {"water": 100, "milk": 200, "coffee beans": 20}},
}

# Initial resources
initial_resources = {"water": 1000, "milk": 1000, "coffee beans": 500}

# Coffee prices (added)
coffee_prices = {
    "espresso": {"small": 2.5, "medium": 2.8, "large": 3.0},
    "latte": {"small": 3.0, "medium": 3.5, "large": 4.0},
    "cappuccino": {"small": 3.0, "medium": 3.5, "large": 4.0},
    "mocha": {"small": 3.0, "medium": 3.5, "large": 4.0},
}

# Function to display resources
def display_resources(resources: Dict[str, float]) -> str:
    return f"Water: {resources['water']}ml, Milk: {resources['milk']}ml, Coffee Beans: {resources['coffee beans']}g"

# Function to refill resources
def refill_resources(resources: Dict[str, float], refill_amount: Dict[str, float]) -> Dict[str, float]:
    for key in refill_amount:
        resources[key] += refill_amount[key]
    return resources

# Function to add custom coffee recipe
def add_custom_coffee_recipe(coffee_type: str, recipe: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    coffee_recipes[coffee_type] = recipe
    return coffee_recipes

# Function to display coffee history (to be expanded as per your project)
def display_history():
    return coffee_history

# Function to make coffee
def coffee_machine(resources: Dict[str, float], coffee_choice: str, size: str, strength: str, price: float) -> Dict[str, float]:
    if coffee_choice in coffee_recipes and size in coffee_recipes[coffee_choice]:
        ingredients = coffee_recipes[coffee_choice][size]
        if resources["water"] >= ingredients["water"] and resources["milk"] >= ingredients["milk"] and resources["coffee beans"] >= ingredients["coffee beans"]:
            resources["water"] -= ingredients["water"]
            resources["milk"] -= ingredients["milk"]
            resources["coffee beans"] -= ingredients["coffee beans"]
            print(f"Making {coffee_choice} ({size}, {strength}) - Price: ${price:.2f}")
        else:
            print("Not enough resources!")
    else:
        print("Invalid coffee choice or size.")
    return resources

# Convert coffee history to a DataFrame
def create_dataset(coffee_history: List[Dict[str, Union[str, float]]]) -> pd.DataFrame:
    df = pd.DataFrame(coffee_history)
    return df

# Encode categorical features (coffee_type, size, strength)
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le_coffee = LabelEncoder()
    le_size = LabelEncoder()
    le_strength = LabelEncoder()
    
    df["coffee_type_encoded"] = le_coffee.fit_transform(df["coffee_type"])
    df["size_encoded"] = le_size.fit_transform(df["size"])
    df["strength_encoded"] = le_strength.fit_transform(df["strength"])
    
    return df, le_coffee, le_size, le_strength

# Prepare features and target for ML
def prepare_data(df: pd.DataFrame):
    X = df[["size_encoded", "strength_encoded"]]
    y = df["coffee_type_encoded"]
    return X, y

# Train the ML model
def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a decision tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# Predict coffee choice based on size and strength
def predict_coffee(model, size: str, strength: str, le_size: LabelEncoder, le_strength: LabelEncoder, le_coffee: LabelEncoder) -> str:
    size_encoded = le_size.transform([size])[0]
    strength_encoded = le_strength.transform([strength])[0]
    
    # Predict the coffee type
    prediction = model.predict([[size_encoded, strength_encoded]])
    coffee_type = le_coffee.inverse_transform(prediction)[0]
    return coffee_type

# Coffee machine workflow with ML integration
def coffee_machine_with_ml(resources: Dict[str, float], coffee_choice: Optional[str] = None, size: Optional[str] = None, strength: Optional[str] = None, price: Optional[float] = None) -> Dict[str, float]:
    print("Welcome to the Coffee Machine with ML!")
    print("Current resources:")
    print(display_resources(resources))

    # If no coffee choice is provided, use ML to predict
    if not coffee_choice:
        # Create dataset from coffee history
        df = create_dataset(coffee_history)
        if not df.empty:
            df, le_coffee, le_size, le_strength = encode_features(df)
            X, y = prepare_data(df)
            model = train_model(X, y)
            
            # Predict coffee choice based on user input for size and strength
            if not size:
                size = input("Choose size (small/medium/large): ").strip().lower()
            if not strength:
                strength = input("Choose strength (weak/normal/strong): ").strip().lower()
            
            coffee_choice = predict_coffee(model, size, strength, le_size, le_strength, le_coffee)
            print(f"\nBased on your preferences, we recommend: {coffee_choice} ({size}, {strength})")
        else:
            print("No historical data available. Please choose a coffee manually.")
            coffee_choice = input("Enter your coffee choice (espresso, latte, cappuccino, mocha): ").strip().lower()
            size = input("Choose size (small/medium/large): ").strip().lower()
            strength = input("Choose strength (weak/normal/strong): ").strip().lower()
    
    # Make the coffee
    if coffee_choice in coffee_recipes and size in coffee_recipes[coffee_choice] and strength in ["weak", "normal", "strong"]:
        price = coffee_prices.get(coffee_choice, {}).get(size, 0.0)
        resources = coffee_machine(resources, coffee_choice, size, strength, price)
    else:
        print("Invalid coffee choice, size, or strength. Please select from the available options.")

    # Show coffee making history
    print("\nCoffee Making History:")
    print(display_history())

    # Refill resources if needed
    refill_choice = input("\nWould you like to refill resources (yes/no)? ").strip().lower()
    if refill_choice == "yes":
        refill_amount = {"water": float(input("Enter amount of water to refill (ml): ")),
                         "milk": float(input("Enter amount of milk to refill (ml): ")),
                         "coffee beans": float(input("Enter amount of coffee beans to refill (g): "))}
        resources = refill_resources(resources, refill_amount)
        print("\nResources after refill:")
        print(display_resources(resources))

    return resources

# Example usage
if __name__ == "__main__":
    resources = initial_resources
    
    # Add custom coffee recipe
    print("Add custom coffee recipe:")
    coffee_recipes = add_custom_coffee_recipe("mocha", {
        "small": {"water": 150, "milk": 100, "coffee beans": 20},
        "medium": {"water": 200, "milk": 150, "coffee beans": 25},
        "large": {"water": 250, "milk": 200, "coffee beans": 30}
    })
    print(f"Updated Coffee Recipes: {coffee_recipes}")
    
    # Let the user make a coffee (with ML recommendation)
    resources = coffee_machine_with_ml(resources)