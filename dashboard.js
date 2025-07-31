import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getFirestore, collection, getDocs, query, orderBy, where } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

// Firebase Config
const firebaseConfig = {
  apiKey: "AIzaSyC__9592DeMcaqFDgVqwRFv9h2Q1cnCGoQ",
  authDomain: "animaldetectionapp-4dd12.firebaseapp.com",
  projectId: "animaldetectionapp-4dd12",
  messagingSenderId: "179378883022",
  appId: "1:179378883022:web:3181a29e013dff210f439f"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

// Wait for login
onAuthStateChanged(auth, async (user) => {
  if (!user) {
    window.location.href = "index.html";
    return;
  }

  console.log("✅ User authenticated:", user.email);
  document.getElementById("userEmail").innerText = user.email;

  try {
    // Query for this user's detections, ordered by timestamp
    const q = query(
      collection(db, "detections"), 
      where("email", "==", user.email),
      orderBy("timestamp", "desc")
    );
    
    console.log("🔍 Querying detections for user:", user.email);
    const snapshot = await getDocs(q);
    
    console.log("📊 Query returned", snapshot.size, "documents");

    const historyTable = document.getElementById("historyTable");
    const accuracyList = [];
    const lossList = [];
    const labelList = [];

    let latest = null;
    let recordCount = 0;

    // Clear existing table content
    historyTable.innerHTML = "";

    snapshot.forEach((doc) => {
      const d = doc.data();
      recordCount++;
      
      console.log(`📝 Processing record ${recordCount}:`, d);

      // Save latest for top section (first record since we're ordering by desc)
      if (!latest) {
        latest = d;
      }

      // Add to table
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${d.timestamp || 'N/A'}</td>
        <td>${d.animal || 'Unknown'}</td>
        <td>${d.emotion || 'Unknown'}</td>
        <td>${d.confidence_animal ? (d.confidence_animal * 100).toFixed(1) + '%' : 'N/A'}</td>
        <td>${d.confidence_emotion ? (d.confidence_emotion * 100).toFixed(1) + '%' : 'N/A'}</td>
        <td>${d.accuracy ? (d.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
        <td>${d.loss ? d.loss.toFixed(3) : 'N/A'}</td>
      `;
      historyTable.appendChild(row);

      // Charts data (reverse order for chronological display)
      if (d.accuracy !== undefined && d.loss !== undefined) {
        accuracyList.unshift((d.accuracy * 100).toFixed(1));
        lossList.unshift(d.loss.toFixed(2));
        labelList.unshift(d.timestamp || `Record ${recordCount}`);
      }
    });

    console.log("📊 Total records processed:", recordCount);

    // Fill latest values
    if (latest) {
      console.log("🎯 Displaying latest detection:", latest);
      document.getElementById("animalName").innerText = latest.animal || 'Unknown';
      document.getElementById("emotionName").innerText = latest.emotion || 'Unknown';
      document.getElementById("animalConf").innerText = latest.confidence_animal ? 
        (latest.confidence_animal * 100).toFixed(1) : 'N/A';
      document.getElementById("emotionConf").innerText = latest.confidence_emotion ? 
        (latest.confidence_emotion * 100).toFixed(1) : 'N/A';
    } else {
      console.log("⚠️ No records found for user");
      // Show placeholder message
      document.getElementById("animalName").innerText = 'No data';
      document.getElementById("emotionName").innerText = 'No data';
      document.getElementById("animalConf").innerText = 'N/A';
      document.getElementById("emotionConf").innerText = 'N/A';
      
      // Add message to table
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="7" style="text-align: center; padding: 20px; color: #666;">No detection records found. Start using the detector to see data here!</td>`;
      historyTable.appendChild(row);
    }

    // Draw line chart only if we have data
    const ctx = document.getElementById("lossAccuracyChart").getContext("2d");
    
    if (accuracyList.length > 0) {
      console.log("📈 Creating chart with", accuracyList.length, "data points");
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: labelList,
          datasets: [
            {
              label: 'Accuracy (%)',
              data: accuracyList,
              borderColor: 'green',
              backgroundColor: 'rgba(0, 255, 0, 0.1)',
              fill: false,
              tension: 0.1
            },
            {
              label: 'Loss',
              data: lossList,
              borderColor: 'red',
              backgroundColor: 'rgba(255, 0, 0, 0.1)',
              fill: false,
              tension: 0.1
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'Model Performance Over Time'
            }
          },
          scales: {
            y: { 
              beginAtZero: true,
              max: 100
            },
            x: {
              display: true,
              title: {
                display: true,
                text: 'Time'
              }
            }
          }
        }
      });
    } else {
      console.log("📈 No data for chart, showing placeholder");
      // Show placeholder chart
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['No Data'],
          datasets: [
            {
              label: 'Accuracy (%)',
              data: [0],
              borderColor: 'green',
              fill: false
            },
            {
              label: 'Loss',
              data: [0],
              borderColor: 'red',
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'No Data Available - Start Using the Detector!'
            }
          },
          scales: {
            y: { beginAtZero: true }
          }
        }
      });
    }

  } catch (error) {
    console.error("❌ Error fetching data:", error);
    
    // Show error message
    const historyTable = document.getElementById("historyTable");
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="7" style="text-align: center; padding: 20px; color: #d32f2f;">Error loading data: ${error.message}</td>`;
    historyTable.appendChild(row);
    
    // Show error in latest section too
    document.getElementById("animalName").innerText = 'Error';
    document.getElementById("emotionName").innerText = 'Error';
    document.getElementById("animalConf").innerText = 'N/A';
    document.getElementById("emotionConf").innerText = 'N/A';
  }
});