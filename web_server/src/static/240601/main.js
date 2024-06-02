let players = {};

function onYouTubeIframeAPIReady() {
    // This function will be called by the YouTube IFrame API when it's ready
}

async function search(q_type) {
    document.getElementById('loading-spinner').style.display = 'block';

    // Get query from the called search bar
    let query = "";
    if (q_type == 'sym') {
        query = document.getElementById('search-query-sym').value;
    } else if (q_type == 'asym') {
        query = document.getElementById('search-query-asym').value;
    }

    // Get channels selected from the sidebar
    const channels = [
        "based_camp",
        "balaji_srinivasan",
        "hormozis",
        "y_combinator",
        "charter_cities_institute",
        "startup_societies_foundation",
        "free_cities_foundation",
        "james_lindsay",
        "jordan_b_peterson"
    ];
    let selected_channels = [];
    for (let i = 0; i < channels.length; i++) {
        if (document.getElementById(channels[i]).classList.contains('selected')) {
            selected_channels.push(channels[i]);
        }
    }
    if (selected_channels.length == 0) {
        selected_channels = channels;
    }
    const channels_string = selected_channels.join(",");

    let response = await fetch(`/search/?query=${query}&channels=${channels_string}&q_type=${q_type}`);
    let data = await response.json();

    document.getElementById('loading-spinner').style.display = 'none';
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results

    // Process results to sort by score and group by videoId
    let segments = Object.keys(data.results).map(idx => {
        const segment = data.results[idx];
        const [videoId, start, end, score] = segment;
        return { videoId, start, end, score };
    });

    let groupedSegments = segments.reduce((acc, segment) => {
        if (!acc[segment.videoId]) {
            acc[segment.videoId] = [];
        }
        acc[segment.videoId].push(segment);
        return acc;
    }, {});

    // Create and append video containers
    Object.keys(groupedSegments).forEach(videoId => {
        const videoContainer = document.createElement("div");
        videoContainer.className = "video-container";

        const headerContainer = document.createElement("div");
        headerContainer.className = "header-container";
        videoContainer.appendChild(headerContainer);

        const toggleEmbedBtn = document.createElement("button");
        toggleEmbedBtn.textContent = "Show/Hide Video";
        headerContainer.appendChild(toggleEmbedBtn);

        const videoTitle = document.createElement("h3");
        videoTitle.textContent = `Video ID: ${videoId}`;
        headerContainer.appendChild(videoTitle);

        let iframe, scoreContainer;
        const contentContainer = document.createElement("div");
        contentContainer.className = "content-container";
        videoContainer.appendChild(contentContainer);

        toggleEmbedBtn.onclick = function () {
            if (!iframe) {
                iframe = document.createElement("div");
                iframe.id = `player-${videoId}`;
                contentContainer.appendChild(iframe);

                players[videoId] = new YT.Player(`player-${videoId}`, {
                    height: '315',
                    width: '560',
                    videoId: videoId,
                    events: {
                        'onReady': onPlayerReady
                    }
                });

                scoreContainer = document.createElement("div");
                scoreContainer.className = "score-container";
                contentContainer.appendChild(scoreContainer);

                groupedSegments[videoId].sort((a, b) => b.score - a.score).forEach(({ start, end, score }) => {
                    const videoLink = `https://www.youtube.com/watch?v=${videoId}&t=${start}s`;

                    let entryContainer = document.createElement("div");
                    entryContainer.className = "entry-container";

                    let controlsContainer = document.createElement("div");
                    controlsContainer.className = "controls-container";

                    let scoreDisplay = document.createElement("div");
                    scoreDisplay.textContent = `Score: ${score.toFixed(3)}`;

                    let copyBtn = document.createElement("button");
                    copyBtn.textContent = "Copy Link";
                    copyBtn.style.position = "relative";
                    copyBtn.onclick = function () {
                        navigator.clipboard.writeText(videoLink).then(() => {
                            let notification = document.createElement("div");
                            notification.textContent = "Copied!";
                            notification.className = "notification";
                            copyBtn.appendChild(notification);
                            setTimeout(() => notification.remove(), 1000);
                        });
                    };

                    let playBtn = document.createElement("button");
                    playBtn.textContent = "Play Timestamp";
                    playBtn.style.position = "relative";
                    playBtn.onclick = function () {
                        playVideoAtTime(videoId, start, toggleEmbedBtn);
                    };

                    controlsContainer.appendChild(playBtn);
                    controlsContainer.appendChild(copyBtn);
                    controlsContainer.appendChild(scoreDisplay);

                    entryContainer.appendChild(controlsContainer);
                    scoreContainer.appendChild(entryContainer);
                });
            } else {
                iframe.remove();
                iframe = null;
                if (scoreContainer) {
                    scoreContainer.remove();
                }
            }
        };

        resultsDiv.appendChild(videoContainer);
    });
}

function handleKeyPressSym(event) {
    if (event.keyCode === 13 || event.which === 13) {
        search('sym');
    }
}

function handleKeyPressAsym(event) {
    if (event.keyCode === 13 || event.which === 13) {
        search('asym');
    }
}

function copySearchURL(qType) {
    const baseUrl = window.location.origin + window.location.pathname;
    const url = new URL(baseUrl);
    const searchCopyBtn = document.getElementById(`search-copy-btn-${qType}`);

    const query = document.getElementById(`search-query-${qType}`).value;
    url.searchParams.set('search', query);

    const channels = document.getElementsByClassName('channel-container');
    let selected_channels = [];
    for (let i = 0; i < channels.length; i++) {
        if (channels[i].classList.contains('selected')) {
            selected_channels.push(channels[i].id);
        }
    }
    if (selected_channels.length > 0) {
        url.searchParams.set('channels', selected_channels.join('+'));
    }
    if (qType == 'asym') {
        url.searchParams.set('asym', 'true');
    }

    navigator.clipboard.writeText(url).then(() => {
        let notification = document.createElement("div");
        notification.textContent = "Copied!";
        notification.className = "notification";
        searchCopyBtn.appendChild(notification);
        setTimeout(() => notification.remove(), 1000);
    });
}

function toggleChannelSelection(event) {
    event.currentTarget.classList.toggle('selected');
}

function playVideoAtTime(videoId, start, toggleEmbedBtn) {
    if (players[videoId]) {
        players[videoId].seekTo(start);
        players[videoId].playVideo();
    } else {
        toggleEmbedBtn.click();
        setTimeout(() => {
            players[videoId].seekTo(start);
            players[videoId].playVideo();
        }, 1000);
    }
}

function onPlayerReady(event) {
    // Player is ready
}

function getSlugFromURL() {
    const urlParams = new URLSearchParams(window.location.search);
    let asym = urlParams.has('asym')
    const query = urlParams.get('search');
    const channels = urlParams.get('channels');
    return [asym, query ? decodeURIComponent(query) : '', channels ? channels : ''];
}

window.onload = function () {
    document.getElementById('search-query-sym').addEventListener('keypress', handleKeyPressSym);
    document.getElementById('search-query-asym').addEventListener('keypress', handleKeyPressAsym);

    const channels = document.getElementsByClassName('channel-container');
    for (let i = 0; i < channels.length; i++) {
        channels[i].addEventListener('click', toggleChannelSelection);
    }

    const slug = getSlugFromURL();
    const [asym, query, channelsString] = slug;
    if (!query) {
        return;
    }

    const re = /[,\+\ ]|%20/
    channelsString.split(re).forEach(channel => {
        const element = document.getElementById(channel);
        if (element) {
            element.classList.add('selected');
        }
    });
    if (!asym) {
        document.getElementById('search-query-sym').value = query;
        search('sym');
    }
    else {
        document.getElementById('search-query-asym').value = query;
        search('asym');
    }
};
