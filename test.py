# Import des librairies
from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app
import os


# assertEqual(a, b) : Vérifie si a est égal à b.
# assertNotEqual(a, b) : Vérifie si a est différent de b.
        
# assertIn(a, b) : Vérifie si a est dans b.
# assertNotIn(a, b) : Vérifie si a n'est pas dans b.
        
# assertIs(a, b) : Vérifie si a est b.
# assertIsNot(a, b) : Vérifie si a n'est pas b.
        
# assertTrue(x) : Vérifie si x est vrai.
# assertFalse(x) : Vérifie si x est faux.
        
# assertIsNone(x) : Vérifie si x est None.
# assertIsNotNone(x) : Vérifie si x n'est pas None.
        
# assertIsInstance(a, b) : Vérifie si a est une instance de b.
# assertNotIsInstance(a, b) : Vérifie si a n'est pas une instance de b.
        
# assertRaises(exc, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc.
# assertRaisesRegex(exc, r, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc et dont le message correspond à l'expression régulière r.


# Tests unitaire de l'environnement de développement
class TestDev(TestCase):

    # Vérifie que les fichiers sont présents
    def test_env(self):
        files = os.listdir()
        self.assertIn("api.py", files)
        self.assertIn("model.pkl", files)
        self.assertIn("test.py", files)
        self.assertIn("requirements.txt", files)

    # Vérifie que les requirements sont présents
    def test_requirements(self):
        with open("requirements.txt") as f:
            requirements = f.read().splitlines()
        self.assertIn("uvicorn==0.27.0", requirements)
        self.assertIn("fastapi==0.109.0", requirements)
        self.assertIn("pydantic-core==2.14.6", requirements)
        self.assertIn("scikit-learn==1.1.2", requirements)

    # Vérifie que le gitignore est présent
    def test_gitignore(self):
        with open(".gitignore") as f:
            gitignore = f.read().splitlines()
        self.assertIn("__pycache__/", gitignore)
        self.assertIn(".vscode/", gitignore)
        self.assertIn("venv_fastapi/", gitignore)

# Création du client de test
client = TestClient(app)

# Tests unitaire de l'API
class TestApi(TestCase):
    
    # Vérifie que l'API est bien lancée
    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello World"})

    # Vérifie que l'API est bien lancée
    def test_hello_you(self):
        response = client.get("/hello_you")
        self.assertEqual(response.status_code, 422)
        response = client.get("/hello_you?name=John")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello John"})

    def test_hello_you_name(self):
        response = client.get("/hello_you/John")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello John"})

    def test_predict(self):
        response = client.post("/predict")
        self.assertEqual(response.status_code, 422)
        response = client.post("/predict", json={
            "age": 58,
            "job": 1,
            "marital": 1,
            "education": 2,
            "default": 0,
            "balance": 2143,
            "housing": 1,
            "loan": 0,
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": 3
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": 0})


# Test du modèle individuellement
class TestModel(TestCase):

    # Vérifie que le modèle est bien présent
    def test_model(self):
        self.assertTrue(os.path.exists("model.pkl"))

    # Vérifie que le modèle est bien chargé
    def test_load_model(self):
        import pickle
        model = pickle.load(open("model.pkl", "rb"))
        self.assertIsNotNone(model)

    # Vérifie que le modèle est bien chargé
    def test_predict(self):
        import pickle
        model = pickle.load(open("model.pkl", "rb"))
        prediction = model.predict([[58, 1, 1, 2, 0, 2143, 1, 0, 1, -1, 0, 3]])
        self.assertEqual(prediction[0], 0)


if __name__ == "__main__":
    main(
        verbosity=2,
        failfast=True
    )
    