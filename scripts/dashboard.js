        // const API_URL = 'http://localhost:8000';
        const API_URL = 'https://chat.momsandkidsworld.com';
        let currentPage = 1;
        const PAGE_SIZE = 10;

        // --- Navigation --- 
        function showSection(sectionId, element) {
            if (element) {
                event.preventDefault();
                document.querySelectorAll('.sidebar ul li').forEach(li => li.classList.remove('active'));
                element.closest('li').classList.add('active');
            }
            document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
            document.getElementById(sectionId + '-section').classList.add('active');

            if (sectionId === 'images') loadImages();
            else if (sectionId === 'products') loadProducts(1);
        }

        // --- Image Management ---
        async function uploadImage() {
            const input = document.getElementById('imageUpload');
            if (input.files.length === 0) return;
            const formData = new FormData();
            for (const file of input.files) formData.append('files', file);

            try {
                const response = await fetch(`${API_URL}/api/images`, { method: 'POST', body: formData });
                if (!response.ok) throw new Error('Upload failed');
                input.value = '';
                loadImages();
            } catch (error) { alert('Error uploading images.'); }
        }

        async function loadImages() {
            try {
                const response = await fetch(`${API_URL}/api/images`);
                const images = await response.json();
                const gallery = document.getElementById('imageGallery');
                gallery.innerHTML = '';
                images.forEach(image => {
                    const item = document.createElement('div');
                    item.className = 'image-item';
                    item.innerHTML = `<img src="${image.image_path}" alt="Image"><button class="btn btn-danger btn-sm delete-btn" onclick="deleteImage(${image.id})"><i class="fas fa-trash-alt"></i></button>`;
                    gallery.appendChild(item);
                });
            } catch (error) { alert('Error loading images.'); }
        }

        async function deleteImage(id) {
            if (!confirm('Are you sure?')) return;
            try {
                await fetch(`${API_URL}/api/images/${id}`, { method: 'DELETE' });
                loadImages();
            } catch (error) { alert('Error deleting image.'); }
        }

        // --- Product Management ---
        function prepareProductModal(product = null) {
            const form = document.getElementById('productForm');
            form.reset();
            document.getElementById('productId').value = product ? product.id : '';
            if (product) {
                document.getElementById('productName').value = product.name;
                document.getElementById('productDescription').value = product.description;
                document.getElementById('productPrice').value = product.price;
                document.getElementById('productCode').value = product.code;
                document.getElementById('marginalPrice').value = product.marginal_price;
                document.getElementById('productLink').value = product.link;
            }
            loadImagesForProductModal(product ? product.images : []);
        }
        
        async function loadImagesForProductModal(productImages = []) {
            try {
                const response = await fetch(`${API_URL}/api/images`);
                const allImages = await response.json();
                const productGallery = document.getElementById('productImageGallery');
                productGallery.innerHTML = '';

                allImages.forEach(image => {
                    const productItem = document.createElement('div');
                    productItem.className = 'image-item';
                    productItem.dataset.imageId = image.id;

                    const existingImage = productImages.find(pi => pi.image_id === image.id);

                    productItem.innerHTML = `
                        <img src="${image.image_path}" alt="Image">
                        <div class="image-options" style="display: none;">
                            <div class="form-check"><input type="checkbox" class="form-check-input" id="cat_${image.id}" ${existingImage?.is_catalogue_image ? 'checked' : ''}><label class="form-check-label" for="cat_${image.id}">Catalogue</label></div>
                            <div class="form-check"><input type="checkbox" class="form-check-input" id="var_${image.id}" ${existingImage?.is_variant_image ? 'checked' : ''}><label class="form-check-label" for="var_${image.id}">Variant</label></div>
                            <div class="form-check"><input type="checkbox" class="form-check-input" id="real_${image.id}" ${existingImage?.is_real_image ? 'checked' : ''}><label class="form-check-label" for="real_${image.id}">Real</label></div>
                            <div class="form-check"><input type="checkbox" class="form-check-input" id="size_${image.id}" ${existingImage?.is_size_related_image ? 'checked' : ''}><label class="form-check-label" for="size_${image.id}">Size</label></div>
                        </div>`;

                    if (existingImage) {
                        productItem.classList.add('selected');
                        productItem.querySelector('.image-options').style.display = 'block';
                    }

                    const optionsContainer = productItem.querySelector('.image-options');
                    optionsContainer.addEventListener('click', (event) => {
                        event.stopPropagation();
                    });

                    productItem.onclick = () => {
                        productItem.classList.toggle('selected');
                        optionsContainer.style.display = productItem.classList.contains('selected') ? 'block' : 'none';
                    };
                    productGallery.appendChild(productItem);
                });
            } catch (error) { console.error('Error loading images for modal:', error); }
        }

        async function saveProduct() {
            const id = document.getElementById('productId').value;
            const images = Array.from(document.querySelectorAll('#productImageGallery .image-item.selected')).map(item => ({
                image_id: parseInt(item.dataset.imageId),
                is_catalogue_image: item.querySelector('[id^=cat_]').checked,
                is_variant_image: item.querySelector('[id^=var_]').checked,
                is_real_image: item.querySelector('[id^=real_]').checked,
                is_size_related_image: item.querySelector('[id^=size_]').checked,
            }));

            const product = {
                name: document.getElementById('productName').value,
                description: document.getElementById('productDescription').value,
                price: parseFloat(document.getElementById('productPrice').value),
                code: document.getElementById('productCode').value,
                marginal_price: parseFloat(document.getElementById('marginalPrice').value),
                link: document.getElementById('productLink').value,
                images: images
            };

            if (!product.name || !product.price || !product.code) { alert('Please fill all required fields.'); return; }

            const url = id ? `${API_URL}/api/products/${id}` : `${API_URL}/api/products`;
            const method = id ? 'PUT' : 'POST';

            try {
                const response = await fetch(url, { method: method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(product) });
                if (!response.ok) throw new Error('Failed to save product');
                $('#productModal').modal('hide');
                loadProducts(id ? currentPage : 1);
            } catch (error) { alert('Error saving product.'); }
        }

        async function loadProducts(page = 1) {
            currentPage = page;
            const params = new URLSearchParams({ page: currentPage, page_size: PAGE_SIZE });
            
            const name = document.getElementById('filterName').value;
            const code = document.getElementById('filterCode').value;
            const minPrice = document.getElementById('filterMinPrice').value;
            const maxPrice = document.getElementById('filterMaxPrice').value;

            if (name) params.append('name', name);
            if (code) params.append('code', code);
            if (minPrice) params.append('min_price', minPrice);
            if (maxPrice) params.append('max_price', maxPrice);

            try {
                const response = await fetch(`${API_URL}/api/products?${params.toString()}`);
                const data = await response.json();
                const tableBody = document.getElementById('productTableBody');
                tableBody.innerHTML = '';

                let products = [];
                let total = 0;

                // Defensive check for API response format
                if (data && data.products) {
                    // New format: { products: [...], total: ... }
                    products = data.products;
                    total = data.total;
                } else if (Array.isArray(data)) {
                    // Fallback for old format: [...]
                    console.warn('Warning: API is returning data in an outdated format. Pagination may not work.');
                    products = data;
                    total = data.length;
                }

                if (products) {
                    products.forEach(product => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${product.name}</td>
                            <td>${product.description ? product.description.substring(0, 50) : ''}...</td>
                            <td>${product.price}</td>
                            <td>${product.marginal_price}</td>
                            <td>${product.code}</td>
                            <td>${(product.images || []).map(img => `<img src="${img.image_path}" width="50" class="img-thumbnail">`).join(' ')}</td>
                            <td>
                                <button class="btn btn-sm btn-primary" data-toggle="modal" data-target="#productModal" onclick='prepareProductModal(${JSON.stringify(product)})'><i class="fas fa-edit"></i></button>
                                <button class="btn btn-sm btn-danger" onclick="deleteProduct(${product.id})"><i class="fas fa-trash-alt"></i></button>
                            </td>`;
                        tableBody.appendChild(row);
                    });
                }
                renderPagination(total);
            } catch (error) {
                console.error("Error loading products:", error);
                alert('Error loading products. Check the console for details.');
            }
        }
        
        function renderPagination(total) {
            const paginationUl = document.getElementById('pagination');
            paginationUl.innerHTML = '';
            const totalPages = Math.ceil(total / PAGE_SIZE);
            if (totalPages <= 1) return;

            for (let i = 1; i <= totalPages; i++) {
                const li = document.createElement('li');
                li.className = `page-item ${i === currentPage ? 'active' : ''}`;
                
                const link = document.createElement('a');
                link.className = 'page-link';
                link.href = '#';
                link.innerText = i;
                link.onclick = (event) => {
                    event.preventDefault();
                    loadProducts(i);
                };

                li.appendChild(link);
                paginationUl.appendChild(li);
            }
        }

        async function deleteProduct(id) {
            if (!confirm('Are you sure?')) return;
            try {
                await fetch(`${API_URL}/api/products/${id}`, { method: 'DELETE' });
                loadProducts(currentPage);
            } catch (error) { alert('Error deleting product.'); }
        }

        document.getElementById('filterForm').onsubmit = (e) => { e.preventDefault(); loadProducts(1); };

        // --- Training ---
        async function generateJson() {
            try {
                const response = await fetch(`${API_URL}/api/generate-json`, { method: 'POST' });
                if (!response.ok) throw new Error('Failed to generate JSON');
                alert('products.json has been generated successfully.');
            } catch (error) {
                console.error('Error generating JSON:', error);
                alert('Error generating JSON file.');
            }
        }

        async function trainModel() {
            try {
                alert('Training started... This may take a while.');
                const response = await fetch(`${API_URL}/api/train`, { method: 'POST' });
                const result = await response.json();
                if (response.ok) {
                    alert('Training complete!\n' + result.output);
                } else {
                    throw new Error(result.detail);
                }
            } catch (error) {
                console.error('Training failed:', error);
                alert('Training failed:\n' + error.message);
            }
        }

        // Initial load
        window.onload = () => {
            document.getElementById('dashboard-section').classList.add('active');
            document.querySelector('.sidebar ul li').classList.add('active');
        };
   