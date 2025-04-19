(function() {
    // Guard clause to prevent re-execution
    if (window.anomalyPlotsRendered) return;
    window.anomalyPlotsRendered = true;

    // Dark Mode Toggle
    const htmlElement = document.documentElement;
    const darkModeToggle = document.getElementById("dark-mode-toggle");
    const moonIcon = document.getElementById("moon-icon");
    const sunIcon = document.getElementById("sun-icon");

    function updateTheme(isDark) {
        if (isDark) {
            htmlElement.classList.add("dark");
            moonIcon.classList.remove("hidden");
            sunIcon.classList.add("hidden");
            document.cookie = "darkMode=enabled; path=/";
        } else {
            htmlElement.classList.remove("dark");
            moonIcon.classList.add("hidden");
            sunIcon.classList.remove("hidden");
            document.cookie = "darkMode=disabled; path=/";
        }
    }

    // Check saved dark mode or system preference
    const savedDarkMode = document.cookie.includes("darkMode=enabled");
    const isDarkMode = savedDarkMode || 
                      (document.cookie.includes("darkMode") === false && 
                       window.matchMedia("(prefers-color-scheme: dark)").matches);
    updateTheme(isDarkMode);

    darkModeToggle.addEventListener("click", () => {
        const isCurrentlyDark = htmlElement.classList.contains("dark");
        updateTheme(!isCurrentlyDark);
        renderPlots();
    });

    // Plotly Charts
    function renderPlots() {
        const dataElement = document.getElementById('anomaly-data');
        if (!dataElement || !dataElement.dataset.anomaly) {
            console.error("Anomaly data element or dataset not found");
            return;
        }
        const data = JSON.parse(dataElement.dataset.anomaly);
        const isDark = htmlElement.classList.contains("dark");
        const bgColor = isDark ? "#1a1a1a" : "white";
        const plotBgColor = isDark ? "#121212" : "white";
        const fontColor = isDark ? "white" : "black";

        // Line Chart
        const lineData = [
            {
                x: data.dates || [],
                y: data.values || [],
                mode: 'lines+markers',
                name: 'Data',
                line: { color: '#1f77b4' }
            },
            {
                x: data.anomalyDates || [],
                y: data.anomalyValues || [],
                mode: 'markers',
                name: 'Anomalies',
                marker: { color: '#ff0000', size: 10, symbol: 'circle' }
            }
        ];

        const lineLayout = {
            title: `Anomaly Detection Results (${data.bestMethod || 'Unknown'})`,
            xaxis: { title: 'Date' },
            yaxis: { title: data.targetCol || 'Value' },
            showlegend: true,
            margin: { t: 50, b: 100, l: 50, r: 50 },
            paper_bgcolor: bgColor,
            plot_bgcolor: plotBgColor,
            font: { color: fontColor }
        };

        Plotly.newPlot('anomaly-plot', lineData, lineLayout);

        // Bar Chart for Method Comparison
        const methods = Object.keys(data.methodResults || {});
        const counts = Object.values(data.methodResults || {});
        const barData = [{
            x: methods,
            y: counts,
            type: 'bar',
            marker: { 
                color: methods.map(m => m === data.bestMethod ? '#22c55e' : '#1f77b4') 
            }
        }];

        const barLayout = {
            title: 'Anomalies Detected by Each Method',
            xaxis: { title: 'Method' },
            yaxis: { title: 'Number of Anomalies' },
            margin: { t: 50, b: 100, l: 50, r: 50 },
            paper_bgcolor: bgColor,
            plot_bgcolor: plotBgColor,
            font: { color: fontColor }
        };

        Plotly.newPlot('method-bar-chart', barData, barLayout);
    }

    // Initial render of plots
    const dataElement = document.getElementById('anomaly-data');
    if (dataElement && dataElement.dataset.anomaly) {
        renderPlots();
    } else {
        console.error("Anomaly data element or dataset not found");
    }
})();