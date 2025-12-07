const dropZone = document.querySelector('.drag-zone');
const fileList = document.querySelector('.file-list');
const fileInput = document.querySelector('#new-file');
const imageUrlInput = document.querySelector('#image-url');
const form = document.querySelector('#ocr-form');
const statusBox = document.querySelector('#status');
const responseBox = document.querySelector('#response');

let selectedFile = null;

function setStatus(message, isError = false) {
	statusBox.textContent = message;
	statusBox.style.color = isError ? 'red' : 'inherit';
}

function renderFileInfo(file) {
	fileList.innerHTML = '';
	if (!file) {
		return;
	}

	const li = document.createElement('li');
	li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
	fileList.appendChild(li);

	if (file.type && file.type !== 'image/png') {
		setStatus('Uwaga: plik nie ma typu image/png, ale zostanie wyslany.', true);
	}
}

function handleFiles(files) {
	if (!files || files.length === 0) {
		selectedFile = null;
		renderFileInfo(null);
		return;
	}

	selectedFile = files[0];
	renderFileInfo(selectedFile);
}

dropZone.addEventListener('dragover', event => {
	event.preventDefault();
	dropZone.classList.add('over');
});

dropZone.addEventListener('dragleave', () => {
	dropZone.classList.remove('over');
});

dropZone.addEventListener('drop', event => {
	event.preventDefault();
	dropZone.classList.remove('over');
	handleFiles(event.dataTransfer.files);
});

fileInput.addEventListener('change', event => {
	handleFiles(event.target.files);
});

form.addEventListener('submit', async event => {
	event.preventDefault();
	responseBox.textContent = '';
	setStatus('');

	// Wysyłamy na ten sam host/port co frontend; nginx proxy przekieruje na API.
	const endpoint = '/api/ocr-table/';
	const imageUrl = (imageUrlInput.value || '').trim();
	const formData = new FormData();

	if (selectedFile) {
		formData.append('image', selectedFile, selectedFile.name || 'image.png');
	} else if (imageUrl) {
		formData.append('url', imageUrl);
	} else {
		setStatus('Dodaj plik PNG lub podaj link do pliku.', true);
		return;
	}

	try {
		setStatus('Wysylanie danych do API...');
		const response = await fetch(endpoint, {
			method: 'POST',
			body: formData
		});

		const responseText = await response.text();
		setStatus(`Status: ${response.status} ${response.statusText}`);
		responseBox.textContent = responseText;
	} catch (error) {
		console.error(error);
		setStatus(`Blad: ${error.message}`, true);
	}
});
