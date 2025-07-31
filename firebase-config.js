// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC__9592DeMcaqFDgVqwRFv9h2Q1cnCGoQ",
  authDomain: "animaldetectionapp-4dd12.firebaseapp.com",
  projectId: "animaldetectionapp-4dd12",
  messagingSenderId: "179378883022",
  appId: "1:179378883022:web:3181a29e013dff210f439f",
  measurementId: "G-8VWJ609ZRX"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);