const dropZone = document.querySelector('.drag-zone')
const fileList = document.querySelector('.file-list')
console.log('first')
function handleFiles(files) {
	fileList.innerHTML = ''

	for (let file of files) {
		const li = document.createElement('li')
		li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`
		fileList.appendChild(li)
	}
}

dropZone.addEventListener('dragover', e => {
	e.preventDefault()
	dropZone.classList.add('over')
})

dropZone.addEventListener('dragleave', () => {
	dropZone.classList.remove('over')
})

dropZone.addEventListener('drop', e => {
	e.preventDefault()
	dropZone.classList.remove('over')

	const files = e.dataTransfer.files
	handleFiles(files)
})