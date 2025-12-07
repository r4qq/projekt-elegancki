const dropZone = document.querySelector('.drag-zone')
const fileList = document.querySelector('.file-list')
const fileInput = document.querySelector('#new-file')
const imageUrlInput = document.querySelector('#image-url')
const form = document.querySelector('#ocr-form')
const statusBox = document.querySelector('#status')
const responseBox = document.querySelector('#response')

let selectedFile = null

function setStatus(text = '') {
	statusBox.innerHTML = ''

	const loading = document.createElement('div')
	loading.classList.add('loading')

	statusBox.appendChild(loading)

	if (text) {
		const info = document.createElement('span')
		info.textContent = ' ' + text
		statusBox.appendChild(info)
	}
}

function clearStatus() {
	statusBox.innerHTML = ''
}

function renderFileInfo(file) {
	fileList.innerHTML = ''
	if (!file) {
		return
	}

	const li = document.createElement('li')
	li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`
	fileList.appendChild(li)

	if (file.type && file.type !== 'image/png') {
		setStatus('Wysyłanie danych...')
	}
}

function handleFiles(files) {
	if (!files || files.length === 0) {
		selectedFile = null
		renderFileInfo(null)
		return
	}

	selectedFile = files[0]
	renderFileInfo(selectedFile)
}

dropZone.addEventListener('dragover', event => {
	event.preventDefault()
	dropZone.classList.add('over')
})

dropZone.addEventListener('dragleave', () => {
	dropZone.classList.remove('over')
})

dropZone.addEventListener('drop', event => {
	event.preventDefault()
	dropZone.classList.remove('over')
	handleFiles(event.dataTransfer.files)
})

fileInput.addEventListener('change', event => {
	handleFiles(event.target.files)
})

form.addEventListener('submit', async event => {
	event.preventDefault()
	responseBox.textContent = ''
	setStatus('Wysyłanie danych do API...')

	const endpoint = '/api/ocr-table/'
	const imageUrl = (imageUrlInput.value || '').trim()
	const formData = new FormData()

	if (selectedFile) {
		formData.append('image', selectedFile, selectedFile.name || 'image.png')
	} else if (imageUrl) {
		formData.append('url', imageUrl)
	} else {
		clearStatus()
		return
	}

	try {
		const response = await fetch(endpoint, {
			method: 'POST',
			body: formData,
		})

		if (!response.ok) {
			throw new Error('HTTP error: ' + response.status)
		}

		clearStatus()

		responseBox.textContent = 'Pomyślnie przesłano dane.'
	} catch (error) {
		clearStatus()
		responseBox.textContent = 'Błąd podczas przesyłania.'
		console.error(error)
	}
})
