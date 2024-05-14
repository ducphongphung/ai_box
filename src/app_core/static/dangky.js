// scripts.js

const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const gallery = document.getElementById('gallery');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  handleFiles(files);
}

function handleFiles(files) {
  files = [...files];
  files.forEach(previewFile);
}

function previewFile(file) {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = function() {
    const img = document.createElement('img');
    img.src = reader.result;
    gallery.appendChild(img);
  }
}
