$(document).ready(() => {
    $("#csv-file-input").on("change", function () {
		const file = this.files[0];
		const reader = new FileReader();
		reader.readAsText(file, "UTF-8");
		
		reader.onload = function (evt) {
			const text = evt.target.result;
			const allRows = text.split(/\r?\n|\r/);
			const variables = [];
			
			let table = '<table id="data-table" class="hover">';
			for (let singleRow = 0; singleRow < allRows.length; singleRow++) {
				if(allRows[singleRow].trim().length === 0) continue;
				
				if (singleRow === 0) {
					table += '<thead>';
					table += '<tr>';
				} else {
					table += '<tr>';
				}
				
				const rowCells = allRows[singleRow].split('|');
				for (let rowCell = 0; rowCell < rowCells.length; rowCell++) {
					let textValue = rowCells[rowCell];
                    if(textValue.startsWith('"')) {
                        textValue = textValue.substring(1, textValue.length-1);
                    }
					if (singleRow === 0) {
						table += '<th>';
						table += textValue;
						variables.push(textValue);
						table += '</th>';
					} else {
						table += '<td>';
						table += textValue;
						table += '</td>';
					}
				}
				
				if (singleRow === 0) {
					table += '</tr>';
					table += '</thead>';
					table += '<tbody>';
				} else {
					table += '</tr>';
				}
			}
			table += '</tbody>';
			table += '</table>';
			
			$('#load-div').remove();
			$('#table-div').append(table);
			$('#data-table').DataTable({
				"scrollX": true
			});
			
			$('#data-table tbody').on('click', 'tr', function () {
				const payload = {};
				$(this).children().each(function(index) {
					payload[variables[index]] = this.innerText;
				});
				$.ajax({
					type: "POST",
					url: "/score",
					dataType: "json", 
					contentType: "application/json; charset=utf-8",
					data: JSON.stringify(payload),
					success: function(prediction) {
						let output = '';
						
						output += `<b>Manufacturing region:</b> `;
						output += prediction.supernode_pairs.manufacturing_region;
						output += '<br>------------------------------------------<br>';
						
						for(let key in prediction.segment_pairs) {
							output += `<b>${key} = </b> ${prediction.segment_pairs[key]}<br>`
						}
						output += '------------------------------------------<br>';
						
						output += `<b>GM cutoffs:</b><br>`;
						for(let i = 0; i < prediction.gm_cutoffs.length; i++) {
							const gm_cutoff = prediction.gm_cutoffs[i];
							output += `${gm_cutoff.toFixed(3)}`;
							if(i != prediction.gm_cutoffs.length - 1) {
								output += ', ';
							}
						}
						
						const content = document.createElement('div');
						content.innerHTML = output;
						
						swal({
							title: 'Success',
							content: content,
							icon: "success",
						})
					},
					error: function(message) {
						swal("Error", message.responseText, "error");
					}
				});
			});
			
			swal("Success", "CSV loaded successfully.", "success");
		}
		
		reader.onerror = function (evt) {
			swal("Error", "CSV could not be loaded.", "error");
		}
	});
});