$(document).ready(function () {
    try {
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        var recognition = new SpeechRecognition();
    }
    catch (e) {
        console.error(e);
    }

    var english_text = $('#english_text')
    const btn_mic = $('#mic-icon')

    /*-----------------------------
      Voice Recognition 
    ------------------------------*/
    recognition.continuous = false;

    recognition.onresult = function (event) {
        var current = event.resultIndex;
        var transcript = event.results[current][0].transcript;
        english_text.val(transcript);
    };

    recognition.onspeechend = function () {
        btn_mic.removeClass('red-icon')
    }

    recognition.onerror = function (event) {
        btn_mic.removeClass('red-icon')
    }

    btn_mic.click(function () {
        if ($(this).hasClass('red-icon')) {
            $(this).removeClass('red-icon')
            recognition.stop();
        } else {
            $(this).addClass('red-icon')
            recognition.start();
        }
    });

    function readOutLoud(message) {
        var speech = new SpeechSynthesisUtterance();

        speech.text = message;
        speech.volume = 1;
        speech.rate = 1;
        speech.pitch = 1;

        window.speechSynthesis.speak(speech);
        // $(this).removeClass('red-icon')
    }

    $('#speaker-icon').on('click', () => {
        $(this).addClass('red-icon')
        readOutLoud(english_text.val())
    })

});