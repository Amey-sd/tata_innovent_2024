document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('click', () => {
        const inner = card.querySelector('.card-inner');
        inner.style.transform = inner.style.transform === 'rotateY(180deg)' ? '' : 'rotateY(180deg)';
    });
});
