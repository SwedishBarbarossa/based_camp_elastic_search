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
    let selected_channels = channels.filter(channel => document.getElementById(channel).classList.contains('selected'));
    if (selected_channels.length === 0) selected_channels = channels;

    const channels_string = selected_channels.join(",");
    let response = await fetch(`/search/?query=${query}&channels=${channels_string}&q_type=${q_type}`);
    let data = await response.json();

    document.getElementById('loading-spinner').style.display = 'none';
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results

    let segments = Object.keys(data.results).map(idx => {
        const segment = data.results[idx];
        const [videoId, start, end, score] = segment;
        return { videoId, start, end, score };
    });

    let groupedSegments = segments.reduce((acc, segment) => {
        if (!acc[segment.videoId]) {
            acc[segment.videoId] = { 'topScore': segment.score, 'segments': [] };
        } else {
            acc[segment.videoId].topScore = Math.max(segment.score, acc[segment.videoId].topScore);
        }

        acc[segment.videoId]['segments'].push(segment);
        return acc;
    }, {});

    // Create and append video containers
    Object.keys(groupedSegments).forEach(videoId => {
        const resultContainer = document.createElement("div");
        resultContainer.className = "result-container";

        const headerContainer = document.createElement("div");
        headerContainer.className = "header-container";
        resultContainer.appendChild(headerContainer);

        const toggleEmbedBtn = document.createElement("button");
        toggleEmbedBtn.textContent = "Show Video";
        headerContainer.appendChild(toggleEmbedBtn);

        let topScore = groupedSegments[videoId].topScore;
        const segmentTitle = document.createElement("h3");
        segmentTitle.textContent = `Top Score: ${topScore.toFixed(3)}`;
        headerContainer.appendChild(segmentTitle);

        const contentContainer = document.createElement("div");
        contentContainer.className = "content-container";
        resultContainer.appendChild(contentContainer);

        const videoContainer = document.createElement("div");
        videoContainer.className = "video-container";

        let embeddedVideoExists = false;
        let embeddedVideoCurrentlyShowing = false;

        let scoreContainer = document.createElement("div");
        scoreContainer.className = "score-container";

        groupedSegments[videoId]['segments'].sort((a, b) => b.score - a.score).forEach(({ start, end, score }) => {
            const videoLink = `https://www.youtube.com/watch?v=${videoId}&t=${start}s`;

            let entryContainer = document.createElement("div");
            entryContainer.className = "entry-container";

            let controlsContainer = document.createElement("div");
            controlsContainer.className = "controls-container";

            let scoreDisplay = document.createElement("div");
            scoreDisplay.textContent = `Score: ${score.toFixed(3)}`;

            let copyBtn = document.createElement("button");
            copyBtn.textContent = "Copy Link";
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
            playBtn.onclick = function () {
                let useTimeout = !embeddedVideoCurrentlyShowing;

                embeddedVideoCurrentlyShowing = toggleVideoDisplay(embeddedVideoExists, videoId, toggleEmbedBtn, videoContainer, contentContainer, true);
                embeddedVideoExists = embeddedVideoExists || embeddedVideoCurrentlyShowing;

                setTimeout(() => {
                    players[videoId].seekTo(start);
                    players[videoId].playVideo();
                }, (useTimeout ? 500 : 0));
            };

            controlsContainer.appendChild(playBtn);
            controlsContainer.appendChild(copyBtn);
            controlsContainer.appendChild(scoreDisplay);

            entryContainer.appendChild(controlsContainer);
            scoreContainer.appendChild(entryContainer);
        });

        contentContainer.appendChild(scoreContainer);

        toggleEmbedBtn.onclick = function () {
            embeddedVideoCurrentlyShowing = toggleVideoDisplay(embeddedVideoExists, videoId, toggleEmbedBtn, videoContainer, contentContainer);
            embeddedVideoExists = embeddedVideoExists || embeddedVideoCurrentlyShowing;
        };

        resultsDiv.appendChild(resultContainer);
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

function toggleVideoDisplay(embeddedVideoExists, videoId, toggleEmbedBtn, videoContainer, contentContainer, forceShow = false) {
    let currentlyShowing = (videoContainer.parentNode !== null);
    let targetShowing = forceShow ? true : (!currentlyShowing);

    if (currentlyShowing !== targetShowing) {
        if (targetShowing) {
            contentContainer.prepend(videoContainer);
            toggleEmbedBtn.textContent = "Hide Video";
        } else {
            videoContainer.remove();
            toggleEmbedBtn.textContent = "Show Video";
        }
    }

    if (!embeddedVideoExists) {
        let iframe = document.createElement("div");
        iframe.id = `player-${videoId}`;
        videoContainer.appendChild(iframe);

        players[videoId] = new YT.Player(`player-${videoId}`, {
            height: '315',
            width: '560',
            videoId: videoId,
            events: {
                'onReady': onPlayerReady
            }
        });
    }

    return targetShowing;
}

function playVideoAtTime(videoId, start) {
    if (players[videoId]) {
        players[videoId].seekTo(start);
        players[videoId].playVideo();
    }
}

function onPlayerReady(event) {
    // Player is ready
}

function getSlugFromURL() {
    const urlParams = new URLSearchParams(window.location.search);
    const asym = urlParams.has('asym');
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

    const re = /[,\+\ ]|%20/;
    channelsString.split(re).forEach(channel => {
        const element = document.getElementById(channel);
        if (element) {
            element.classList.add('selected');
        }
    });
    if (!asym) {
        document.getElementById('search-query-sym').value = query;
        search('sym');
    } else {
        document.getElementById('search-query-asym').value = query;
        search('asym');
    }
};
