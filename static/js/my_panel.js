/***
 * @Author      TheBetterKong
 * @Description 与 pannel 相关代码
 * @Date        2010/06/24
 */

// pFunc1 根据下拉框展示图片
function changeImage() {
    var styleFileImage = 'img/style/' + document.getElementById('style').value + '.jpg';
    console.log(document.getElementById('style').value);
    console.log(styleFileImage);
    document.getElementById('imageShow').src = "../static/" + styleFileImage;
}

// pFunc2 展示上传的 content 图片
function changeContentImage() {
    function getObjectURL(file) {
        var url = null;
        if (window.createObjcectURL !== undefined) {
            url = window.createOjcectURL(file);
        } else if (window.URL !== undefined) {
            url = window.URL.createObjectURL(file);
        } else if (window.webkitURL !== undefined) {
            url = window.webkitURL.createObjectURL(file);
        }
        return url;
    }
    var fk = document.getElementById("pic");
    var contentimage = fk.files[0];
    var maxSize = 1048576;                  // 设置图片最大 1MB
    var fileSize = contentimage.size;               // 获取上传的文件大小
    if (parseInt(fileSize) >= parseInt(maxSize)) {
        alert("服务器性能有限，请不要上传超过 1M 的图片！")
    }else if(fk.value.indexOf("jpg") !== -1 || fk.value.indexOf("JPG") !== -1 || fk.value.indexOf("jpeg") !== -1
        || fk.value.indexOf("png") !== -1 || fk.value.indexOf("PNG") !== -1) {
        var objURL = getObjectURL(contentimage);    // 这里的 objURL 就是 input file 的真实路径
        document.getElementById('contentimageShow').src = objURL;
    } else {
        alert("您未上传文件，或者您上传文件类型有误！！！\n" + "注意：仅支持 png，jpg，jpeg 格式的图片。")
    }
}

// pFunc3 生成风格转换结果图片
function showImage() {
    // 与服务器交互
    // var style = $("#style").val();                                  // 获取风格
    //
    // var animateimg = $("#pic").val();                               // 获取上传的图片名 带//
    // var imgarr = animateimg.split('\\');                   // 分割
    // var myimg = imgarr[imgarr.length - 1];                          // 去掉 // 获取图片名
    // var houzui = myimg.lastIndexOf('.');                            // 获取 . 出现的位置
    // var ext = myimg.substring(houzui, myimg.length).toUpperCase();  // 切割 . 获取文件后缀
    //
    // var file = $('#pic').get(0).files[0];   // 获取上传的文件
    // var fileSize = file.size;               // 获取上传的文件大小
    // var maxSize = 1048576;                  // 最大 1MB
    //
    // // 处理
    // if (ext !== '.png' && ext !== '.PNG' && ext !== '.jpg' && ext !== '.JPG' && ext !== '.jpeg') {
    //     parent.layer.msg('文件类型错误,请上传图片类型');
    //     return false;
    // } else if (parseInt(fileSize) >= parseInt(maxSize)) {
    //     parent.layer.msg('上传的文件不能超过 1MB');
    //     return false;
    // } else {
    //     var data = new FormData();
    //     data.append("style", style);
    //     data.append("pic", file)
    //     $.ajax({
    //         url: '/transform',
    //         type: 'POST',
    //         data: data,
    //         dataType: 'JSON',
    //         cache: false,
    //         processData: false,
    //         contentType: false
    //     }).done(function (ret) {
    //         if (!ret['isSuccess']) {
    //             layer.msg(ret['status']);
    //         }
    //     });
    //     return false;
    // }
}
