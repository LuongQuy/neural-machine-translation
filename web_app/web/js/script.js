$(document).ready(() => {

    $("#btn-translate").on("click", async () => {
        api = "http://127.0.0.1:5000/translate/"
        url = api + $('#english_text').val() + "/" + $("#attention_type").val() + '/' + $("#beam_width").val();

        await fetch(url, {
            method: "GET",
            headers: {
                "Content-Type": "text/plain;charset=UTF-8"
            },
        })
            .then(response => response.json())
            .then(response => {
                data = response.data
                // console.log(data)
                $('#vietnamese_text_gready').val(data['gready'])
                $('#vietnamese_text_beam').val(data['beam'].join('\n'))
            })
            .catch(error => {
                console.log(error);
            });
    })
});