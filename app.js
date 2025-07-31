// ✅ USE: Firebase v10+ module syntax
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

// ✅ Your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyC__9592DeMcaqFDgVqwRFv9h2Q1cnCGoQ",
  authDomain: "animaldetectionapp-4dd12.firebaseapp.com",
  projectId: "animaldetectionapp-4dd12",
  messagingSenderId: "179378883022",
  appId: "1:179378883022:web:3181a29e013dff210f439f"
};

// ✅ Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// ✅ Handle login form
const loginForm = document.getElementById("loginForm");

loginForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      console.log("✅ Login successful");
      // Optionally save user email to a file/backend here
      window.location.href = "dashboard.html";
    })
    .catch((error) => {
      console.error("❌ Login error:", error.message);
      alert(error.message);
    });
});
