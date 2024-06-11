let players = {};

const channels = [ /* [Channel ID, Channel Name, Channel Type] */
    ["based_camp", "Based Camp", "ideology"],
    ["balaji_srinivasan", "Balaji Srinivasan", "misc"],
    ["hormozis", "Hormozis", "business"],
    ["y_combinator", "Y Combinator", "business"],
    ["charter_cities_institute", "Charter Cities Institute", "charter_cities"],
    ["startup_societies_foundation", "Startup Societies Foundation", "charter_cities"],
    ["free_cities_foundation", "Free Cities Foundation", "charter_cities"],
    ["james_lindsay", "James Lindsay", "ideology"],
    ["jordan_b_peterson", "Jordan B Peterson", "ideology"],
    ["chris_williamson", "Chris Williamson", "misc"],
    ["numberphile", "Numberphile", "science"],
    ["computerphile", "Computerphile", "science"],
    ["ted", "TED", "misc"],
    ["ryan_chapman", "Ryan Chapman", "ideology"],
    ["veritasium", "Veritasium", "science"],
];

function buildChannelList() {
    const channelList = document.getElementById("channels-container");
    const sortedChannels = channels.sort((a, b) => a[1].localeCompare(b[1])).sort((a, b) => a[2].localeCompare(b[2]));;
    for (let i = 0; i < channels.length; i++) {
        const channel = sortedChannels[i];
        const channelElement = document.createElement("div");
        let baseClassName = "channel-container";
        channelElement.className = baseClassName;
        channelElement.id = channel[0];
        const channelType = document.createElement("span");
        channelType.className = `channel-type ${channel[2]}`;
        channelElement.appendChild(channelType);
        const channelLabel = document.createElement("label");
        channelLabel.htmlFor = channel[0];
        channelLabel.textContent = channel[1];
        channelLabel.className = "channel-label";
        channelElement.appendChild(channelLabel);
        channelList.appendChild(channelElement);
    }
}

function writeStrToClipboard(str, btn) {
    navigator.clipboard.writeText(str).then(() => {
        let notification = document.createElement("div");
        notification.textContent = "Copied!";
        notification.className = "notification";
        btn.appendChild(notification);
        setTimeout(() => notification.remove(), 1000);
    });
}

function onYouTubeIframeAPIReady() {
    // This function will be called by the YouTube IFrame API when it's ready
}

function createVideoGroupPill(videoId, segmentGroup) {
    const resultContainer = document.createElement("div");
    resultContainer.className = "result-container";

    const headerContainer = document.createElement("div");
    headerContainer.className = "header-container";
    resultContainer.appendChild(headerContainer);

    const headerContent = document.createElement("div");
    headerContent.className = "header-content";
    headerContainer.appendChild(headerContent);

    const toggleEmbedBtn = document.createElement("button");
    toggleEmbedBtn.textContent = "Show Video";
    headerContent.appendChild(toggleEmbedBtn);

    let topScore = segmentGroup.topScore;
    const segmentTitle = document.createElement("h3");
    segmentTitle.textContent = `Top Score: ${topScore.toFixed(3)}`;
    headerContent.appendChild(segmentTitle);

    const contentContainer = document.createElement("div");
    contentContainer.className = "content-container";
    resultContainer.appendChild(contentContainer);

    const videoContainer = document.createElement("div");
    videoContainer.className = "video-container";

    let embeddedVideoExists = false;
    let embeddedVideoCurrentlyShowing = false;

    let scoreContainer = document.createElement("div");
    scoreContainer.className = "score-container";

    segmentGroup['segments'].sort((a, b) => b.score - a.score).forEach(({ start, end, score }) => {
        const videoLink = `https://www.youtube.com/watch?v=${videoId}&t=${start}s`;

        let entryContainer = document.createElement("div");
        entryContainer.className = "entry-container";

        let controlsContainer = document.createElement("div");
        controlsContainer.className = "controls-container";

        let scoreDisplay = document.createElement("div");
        scoreDisplay.textContent = `Score: ${score.toFixed(3)}`;
        scoreDisplay.className = "timestamp-score";

        let copyBtn = document.createElement("button");
        copyBtn.textContent = "Copy Link";
        copyBtn.onclick = () => writeStrToClipboard(videoLink, copyBtn);

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

    return resultContainer;
}

function getSearchType() {
    const toggleElements = document.getElementsByName('search-type');
    for (let i = 0; i < toggleElements.length; i++) {
        if (toggleElements[i].checked) return toggleElements[i].value;
    }
    return 'err';
}

async function search() {
    clearResults();
    document.getElementById('loading-spinner').style.display = 'block';
    document.getElementById('error-message').style.display = 'none';

    // Get query from the search bar
    let query = document.getElementById('search-field').value;

    // Get type of search
    let q_type = getSearchType();
    if (q_type == 'err') {
        document.getElementById('loading-spinner').style.display = 'none';
        document.getElementById('error-message').style.display = 'block';
        return;
    }

    // Get channels selected from the sidebar
    const channel_ids = channels.map(channel => channel[0]);
    let selected_channels = channel_ids.filter(channel => document.getElementById(channel).classList.contains('selected'));
    if (selected_channels.length === 0) selected_channels = channel_ids;

    // Hit API
    const channels_string = selected_channels.join(",");
    let response = await fetch(`/search/?query=${query}&channels=${channels_string}&q_type=${q_type}`);

    // If response is not ok, show 
    if (!response.ok) {
        document.getElementById('loading-spinner').style.display = 'none';
        document.getElementById('error-message').style.display = 'block';
        return;
    }

    let data = await response.json();

    // Parse results
    let segments = Object.keys(data.results).map(idx => {
        const segment = data.results[idx];
        const [videoId, start, end, score] = segment;
        return { videoId, start, end, score };
    });
    
    // Group segments by video id
    let groupedSegments = segments.reduce((acc, segment) => {
        if (!acc[segment.videoId]) {
            acc[segment.videoId] = { 'topScore': segment.score, 'segments': [] };
        } else {
            acc[segment.videoId].topScore = Math.max(segment.score, acc[segment.videoId].topScore);
        }
        
        acc[segment.videoId]['segments'].push(segment);
        return acc;
    }, {});
    
    // Hide spinner
    document.getElementById('loading-spinner').style.display = 'none';
    
    // Create and append video containers
    let resultsDiv = document.getElementById('results');
    Object.keys(groupedSegments).forEach(videoId => {
        resultsDiv.appendChild(createVideoGroupPill(videoId, groupedSegments[videoId]));
    });
}

function clearResults() {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results
}

function handleSeachEnter(event) {
    if (event.keyCode === 13 || event.which === 13) search();
}


function copySearchURL() {
    const baseUrl = window.location.origin + window.location.pathname;
    const url = new URL(baseUrl);
    const searchCopyBtn = document.getElementById(`search-copy-btn`);

    const query = document.getElementById(`search-field`).value;
    url.searchParams.set('search', query);

    const channels_element = document.getElementsByClassName('channel-container');
    let selected_channels = [];
    for (let i = 0; i < channels_element.length; i++) {
        if (channels_element[i].classList.contains('selected')) {
            selected_channels.push(channels_element[i].id);
        }
    }
    if (selected_channels.length > 0) {
        url.searchParams.set('channels', selected_channels.join('+'));
    }
    const q_type = getSearchType();
    if (q_type === 'asym') url.searchParams.set('asym', 'true');

    writeStrToClipboard(url, searchCopyBtn);
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
    const slug_channels = urlParams.get('channels');
    return [asym, query ? decodeURIComponent(query) : '', slug_channels ? slug_channels : ''];
}

async function addDonationButton() {
    // get donation URL
    const res = await fetch('/donate_url/');
    if (res.ok) {
        const res_text = await res.json();
        const url = res_text.donate_url;
        const message = res_text.message;
        
        // make container
        const container = document.createElement('div');
        container.id = 'donation-container';
        container.className = 'donation-container';
        
        // add text
        const text = document.createElement('div');
        text.textContent = message;
        text.className = 'donation-text';
        container.appendChild(text);

        // add button element
        const button = document.createElement('button');
        button.id = 'donate-button';
        button.textContent = 'Donate';
        button.className = 'donation-button';
        button.onclick = () => window.open(url, '_blank');
        container.appendChild(button);

        const sidebar = document.getElementById('right-sidebar');
        sidebar.appendChild(container);
    }
}

window.onload = function () {
    buildChannelList();
    addDonationButton();
    document.getElementById('search-field').addEventListener('keypress', handleSeachEnter);
    
    /* prevent hover on touch */
    document.body.addEventListener('touchstart', () => {
        try { // prevent exception on browsers not supporting DOM styleSheets properly
            for (var style in document.styleSheets) {
                var styleSheet = document.styleSheets[style];
                if (!styleSheet.cssRules) continue;
        
                for (var rule = styleSheet.cssRules.length - 1; rule >= 0; rule--) {
                if (!styleSheet.cssRules[rule].selectorText) continue;
        
                if (styleSheet.cssRules[rule].selectorText.match(':hover')) {
                    styleSheet.deleteRule(rule);
                }
                }
            }
            } catch (ex) {}
    });

    const channels_element = document.getElementsByClassName('channel-container');
    for (let i = 0; i < channels_element.length; i++) {
        channels_element[i].addEventListener('click', toggleChannelSelection);
    }

    const slug = getSlugFromURL();
    const [asym, query, channelsString] = slug;
    if (!query) return;

    if (asym) document.getElementById('asym-search').checked = true;

    const re = /[,\+\ ]|%20/;
    channelsString.split(re).forEach(channel => {
        const element = document.getElementById(channel);
        if (element) element.classList.add('selected');
    });
    document.getElementById('search-field').value = query;
    search();
};
